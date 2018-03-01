#include "state_space_gaussian_model_manager.h"
#include "model_manager.h"
#include "utils.h"
#include "r_interface/prior_specification.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

namespace BOOM {
namespace bsts {

StateSpaceModelBase * GaussianModelManagerBase::CreateModel(
    SEXP r_data_list,
    SEXP r_state_specification,
    SEXP r_prior,
    SEXP r_options,
    Vector *final_state,
    bool save_state_contribution,
    bool save_prediction_errors,
    RListIoManager *io_manager) {
  StateSpaceModelBase *model = ModelManager::CreateModel(
      r_data_list,
      r_state_specification,
      r_prior,
      r_options,
      final_state,
      save_state_contribution,
      save_prediction_errors,
      io_manager);

  // It is only possible to compute log likelihood for Gaussian models.
  io_manager->add_list_element(
      new BOOM::NativeUnivariateListElement(
          new LogLikelihoodCallback(model),
          "log.likelihood",
          nullptr));
  return model;
}

StateSpaceModel * StateSpaceModelManager::CreateObservationModel(
    SEXP r_data_list,
    SEXP r_prior,
    SEXP r_options,
    RListIoManager *io_manager) {
  model_.reset(new StateSpaceModel);
  // If the model is being created from scratch for the purpose of
  // learning, then r_data_list must be supplied.  If the model is
  // being created from an existing R object then we want to defer
  // adding data until later, because the data may be stored
  // differently or we may want to substitute a different training
  // data set.
  if (!Rf_isNull(r_data_list)) {
    if (Rf_inherits(r_data_list, "bsts")) {
      AddDataFromBstsObject(r_data_list);
    } else {
      AddDataFromList(r_data_list);
    }
  }

  // If the model is begin created from scratch, then we need a prior
  // here.  If the model is being created for purposes other than MCMC
  // then we can allow the caller to skip the prior.
  if (!Rf_isNull(r_prior)) {
    if (!Rf_inherits(r_prior, "SdPrior")) {
      report_error("Prior must inherit from SdPrior.");
    }
    ZeroMeanGaussianModel *observation_model = model_->observation_model();
    RInterface::SdPrior sigma_prior(r_prior);
    Ptr<ZeroMeanGaussianConjSampler> sigma_sampler(
        new ZeroMeanGaussianConjSampler(
            observation_model,
            sigma_prior.prior_df(),
            sigma_prior.prior_guess()));
    sigma_sampler->set_sigma_upper_limit(sigma_prior.upper_limit());
    observation_model->set_method(sigma_sampler);

    Ptr<StateSpacePosteriorSampler> sampler(
        new StateSpacePosteriorSampler(model_.get()));
    model_->set_method(sampler);
  }

  // Make the io_manager aware of the model parameters.
  io_manager->add_list_element(new StandardDeviationListElement(
      model_->observation_model()->Sigsq_prm(),
      "sigma.obs"));

  return model_.get();
}

void StateSpaceModelManager::AddDataFromBstsObject(SEXP r_bsts_object) {
  SEXP r_original_series = getListElement(r_bsts_object, "original.series");
  UnpackTimestampInfo(r_bsts_object);
  AddData(ToBoomVector(r_original_series),
          IsObserved(r_original_series));
}

void StateSpaceModelManager::AddDataFromList(SEXP r_data_list) {
  UnpackTimestampInfo(r_data_list);
  AddData(ToBoomVector(getListElement(r_data_list, "response")),
          ToVectorBool(getListElement(r_data_list, "response.is.observed")));
}

int StateSpaceModelManager::UnpackForecastData(SEXP r_prediction_data) {
  forecast_horizon_ = Rf_asInteger(getListElement(
      r_prediction_data, "horizon"));
  return forecast_horizon_;
}

Vector StateSpaceModelManager::SimulateForecast(const Vector &final_state) {
  return model_->simulate_forecast(rng(), forecast_horizon_, final_state);
}

void StateSpaceModelManager::AddData(
    const Vector &response,
    const std::vector<bool> &response_is_observed) {
  if (!response_is_observed.empty()
      && (response.size() != response_is_observed.size())) {
    report_error("Vectors do not match in StateSpaceModelManager::AddData.");
  }
  std::vector<Ptr<StateSpace::MultiplexedDoubleData>> data;
  for (int i = 0; i < NumberOfTimePoints(); ++i) {
    data.push_back(new StateSpace::MultiplexedDoubleData);
  }
  for (int i = 0; i < response.size(); ++i) {
    NEW(DoubleData, observation)(response[i]);
    if (!response_is_observed.empty() && !response_is_observed[i]) {
      observation->set_missing_status(Data::completely_missing);
    }
    data[TimestampMapping(i)]->add_data(observation);
  }
  for (int i = 0; i < NumberOfTimePoints(); ++i) {
    if (data[i]->all_missing()) {
      data[i]->set_missing_status(Data::completely_missing);
    }
    model_->add_data(data[i]);
  }
}

HoldoutErrorSampler StateSpaceModelManager::CreateHoldoutSampler(
    SEXP r_bsts_object,
    int cutpoint,
    Matrix *prediction_error_output) {
  RListIoManager io_manager;
  Ptr<StateSpaceModel> model = static_cast<StateSpaceModel *>(CreateModel(
      R_NilValue,
      getListElement(r_bsts_object, "state.specification"),
      getListElement(r_bsts_object, "prior"),
      getListElement(r_bsts_object, "model.options"),
      nullptr,
      false,
      true,
      &io_manager));
  AddDataFromBstsObject(r_bsts_object);

  std::vector<Ptr<StateSpace::MultiplexedDoubleData>> data = model->dat();
  model_->clear_data();
  for (int i = 0; i <= cutpoint; ++i) {
    model_->add_data(data[i]);
  }
  Vector holdout_data;
  for (int i = cutpoint + 1; i < data.size(); ++i) {
    Ptr<StateSpace::MultiplexedDoubleData> data_point = data[i];
    for (int j = 0; j < data[i]->total_sample_size(); ++j) {
      holdout_data.push_back(data[i]->double_data(j).value());
    }
  }
  int niter = Rf_asInteger(getListElement(r_bsts_object, "niter"));
  return HoldoutErrorSampler(new StateSpaceModelPredictionErrorSampler(
      model, holdout_data, niter, prediction_error_output));
}


StateSpaceModelPredictionErrorSampler::StateSpaceModelPredictionErrorSampler(
    const Ptr<StateSpaceModel> &model,
    const Vector &holdout_data,
    int niter,
    Matrix *errors)
    : model_(model),
      holdout_data_(holdout_data),
      niter_(niter),
      errors_(errors)
{}

void StateSpaceModelPredictionErrorSampler::sample_holdout_prediction_errors() {
  model_->sample_posterior();
  errors_->resize(niter_, model_->time_dimension() + holdout_data_.size());
  for (int i = 0; i < niter_; ++i) {
    model_->sample_posterior();
    Vector error_sim = model_->one_step_prediction_errors();
    error_sim.concat(model_->one_step_holdout_prediction_errors(
        holdout_data_, model_->final_state()));
    errors_->row(i) = error_sim;
  }
}

}  // namespace bsts
}  // namespace BOOM

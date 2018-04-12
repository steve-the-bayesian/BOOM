// Copyright 2018 Google Inc. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#include "state_space_regression_model_manager.h"
#include "state_space_gaussian_model_manager.h"
#include "utils.h"

#include "r_interface/list_io.hpp"
#include "r_interface/prior_specification.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabDaRegressionSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
namespace bsts {

namespace {
typedef StateSpaceRegressionModelManager SSRMF;
}  // namespace

StateSpaceRegressionModelManager::StateSpaceRegressionModelManager()
    : predictor_dimension_(-1) {}

StateSpaceRegressionModel * SSRMF::CreateObservationModel(
    SEXP r_data_list,
    SEXP r_prior,
    SEXP r_options,
    RListIoManager *io_manager) {
  Matrix predictors;
  Vector response;
  std::vector<bool> response_is_observed;
  if (!Rf_isNull(r_data_list)) {
    if (Rf_inherits(r_data_list, "bsts")) {
      predictors = ToBoomMatrix(getListElement(r_data_list, "predictors"));
      SEXP r_response = getListElement(r_data_list, "original.series");
      response = ToBoomVector(r_response);
      response_is_observed = IsObserved(r_response);
    } else {
      // If we were passed data from R then use it to build the model.
      predictors = ToBoomMatrix(getListElement(r_data_list, "predictors"));
      response = ToBoomVector(getListElement(r_data_list, "response"));
      response_is_observed = ToVectorBool(getListElement(
          r_data_list, "response.is.observed"));
    }
    UnpackTimestampInfo(r_data_list);
    if (TimestampsAreTrivial()) {
      model_.reset(new StateSpaceRegressionModel(
          response,
          predictors,
          response_is_observed));
    } else {
      // timestamps are non-trivial.
      model_.reset(new StateSpaceRegressionModel(ncol(predictors)));
      std::vector<Ptr<StateSpace::MultiplexedRegressionData>> data;
      data.reserve(NumberOfTimePoints());
      for (int i = 0; i < NumberOfTimePoints(); ++i) {
        data.push_back(new StateSpace::MultiplexedRegressionData);
      }
      for (int i = 0; i < response.size(); ++i) {
        NEW(RegressionData, observation)(response[i], predictors.row(i));
        if (!response_is_observed[i]) {
          observation->set_missing_status(Data::completely_missing);
        }
        data[TimestampMapping(i)]->add_data(observation);
      }
      for (int i = 0; i < NumberOfTimePoints(); ++i) {
        if (data[i]->observed_sample_size() == 0) {
          data[i]->set_missing_status(Data::completely_missing);
        }
        model_->add_multiplexed_data(data[i]);
      }
    }
  } else {
    // No data was passed from R, so build the model from its default
    // constructor.  We need to know the dimension of the predictors.
    if (predictor_dimension_ < 0) {
      report_error("If r_data_list is not passed, you must call "
                   "SetPredictorDimension before calling "
                   "CreateObservationModel.");
    }
    model_.reset(new StateSpaceRegressionModel(predictor_dimension_));
  }

  // A NULL r_prior signals that no posterior sampler is needed.
  if (!Rf_isNull(r_prior)) {
    SetRegressionSampler(r_prior, r_options);
    Ptr<StateSpacePosteriorSampler> sampler(
        new StateSpacePosteriorSampler(model_.get()));
    if (!Rf_isNull(r_options)
        && !Rf_asLogical(getListElement(r_options, "enable.threads"))) {
      sampler->disable_threads();
    }
    model_->set_method(sampler);
  }

  // Make the io_manager aware of the model parameters.
  Ptr<RegressionModel> regression(model_->regression_model());
  io_manager->add_list_element(
      new GlmCoefsListElement(regression->coef_prm(), "coefficients"));
  io_manager->add_list_element(
      new StandardDeviationListElement(regression->Sigsq_prm(),
                                       "sigma.obs"));
  return model_.get();
}

void SSRMF::AddDataFromBstsObject(SEXP r_bsts_object) {
  AddData(ToBoomVector(getListElement(r_bsts_object, "original.series")),
          ToBoomMatrix(getListElement(r_bsts_object, "predictors")),
          IsObserved(getListElement(r_bsts_object, "original.series")));
}

void SSRMF::AddDataFromList(SEXP r_data_list) {
  AddData(ToBoomVector(getListElement(r_data_list, "original.series")),
          ToBoomMatrix(getListElement(r_data_list, "predictors")),
          ToVectorBool(getListElement(r_data_list,
                                      "response.is.observed")));
}

int SSRMF::UnpackForecastData(SEXP r_prediction_data) {
  forecast_predictors_ = ToBoomMatrix(getListElement(
      r_prediction_data, "predictors"));
  UnpackForecastTimestamps(r_prediction_data);
  return forecast_predictors_.nrow();
}

Vector SSRMF::SimulateForecast(const Vector &final_state) {
  if (ForecastTimestamps().empty()) {
    return model_->simulate_forecast(rng(), forecast_predictors_, final_state);
  } else {
    return model_->simulate_multiplex_forecast(rng(),
                                               forecast_predictors_,
                                               final_state,
                                               ForecastTimestamps());
  }
}

void SSRMF::SetRegressionSampler(SEXP r_regression_prior,
                                 SEXP r_options) {
  // If either the prior object or the bma method is NULL then take
  // that as a signal the model is not being specified for the
  // purposes of MCMC, and bail out.
  if (Rf_isNull(r_regression_prior)
      || Rf_isNull(r_options)
      || Rf_isNull(getListElement(r_options, "bma.method"))) {
    return;
  }
  std::string bma_method = BOOM::ToString(getListElement(
      r_options, "bma.method"));
  if (bma_method == "SSVS") {
    SetSsvsRegressionSampler(r_regression_prior);
  } else if (bma_method == "ODA") {
    SetOdaRegressionSampler(r_regression_prior, r_options);
  } else {
    std::ostringstream err;
    err << "Unrecognized value of bma_method: " << bma_method;
    BOOM::report_error(err.str());
  }
}

void SSRMF::SetSsvsRegressionSampler(SEXP r_regression_prior) {
  BOOM::RInterface::RegressionConjugateSpikeSlabPrior prior(
      r_regression_prior, model_->regression_model()->Sigsq_prm());
  DropUnforcedCoefficients(model_->regression_model(),
                           prior.prior_inclusion_probabilities());
  Ptr<BregVsSampler> sampler(new BregVsSampler(
      model_->regression_model().get(),
      prior.slab(),
      prior.siginv_prior(),
      prior.spike()));
  sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
  int max_flips = prior.max_flips();
  if (max_flips > 0) {
    sampler->limit_model_selection(max_flips);
  }
  model_->regression_model()->set_method(sampler);
}

void SSRMF::SetOdaRegressionSampler(SEXP r_regression_prior,
                                    SEXP r_options) {
  SEXP r_oda_options = getListElement(r_options, "oda.options");
  BOOM::RInterface::IndependentRegressionSpikeSlabPrior prior(
      r_regression_prior, model_->regression_model()->Sigsq_prm());
  double eigenvalue_fudge_factor = 0.001;
  double fallback_probability = 0.0;
  if (!Rf_isNull(r_oda_options)) {
    eigenvalue_fudge_factor = Rf_asReal(
        getListElement(r_oda_options, "eigenvalue.fudge.factor"));
    fallback_probability = Rf_asReal(
        getListElement(r_oda_options, "fallback.probability"));
  }
  Ptr<SpikeSlabDaRegressionSampler> sampler(
      new SpikeSlabDaRegressionSampler(
          model_->regression_model().get(),
          prior.slab(),
          prior.siginv_prior(),
          prior.prior_inclusion_probabilities(),
          eigenvalue_fudge_factor,
          fallback_probability));
  sampler->set_sigma_upper_limit(prior.sigma_upper_limit());
  DropUnforcedCoefficients(model_->regression_model(),
                           prior.prior_inclusion_probabilities());
  model_->regression_model()->set_method(sampler);
}

void StateSpaceRegressionModelManager::SetPredictorDimension(int xdim) {
  predictor_dimension_ = xdim;
}

void StateSpaceRegressionModelManager::AddData(
    const Vector &response,
    const Matrix &predictors,
    const std::vector<bool> &response_is_observed) {
  if (nrow(predictors) != response.size()
      || response_is_observed.size() != response.size()) {
    std::ostringstream err;
    err << "Argument sizes do not match in "
        << "StateSpaceRegressionModelManager::AddData" << endl
        << "nrow(predictors) = " << nrow(predictors) << endl
        << "response.size()  = " << response.size() << endl
        << "observed.size()  = " << response_is_observed.size();
    report_error(err.str());
  }

  for (int i = 0; i < response.size(); ++i) {
    Ptr<RegressionData> dp(new RegressionData(response[i], predictors.row(i)));
    if (!response_is_observed[i]) {
      dp->set_missing_status(Data::partly_missing);
    }
    model_->add_regression_data(dp);
  }
}

namespace {
typedef StateSpaceRegressionHoldoutErrorSampler ErrorSampler;
}  // namespace

void ErrorSampler::sample_holdout_prediction_errors() {
  model_->sample_posterior();
  errors_->resize(niter_, model_->time_dimension() + holdout_responses_.size());
  for (int i = 0; i < niter_; ++i) {
    model_->sample_posterior();
    Vector all_errors = model_->one_step_prediction_errors();
    all_errors.concat(model_->one_step_holdout_prediction_errors(
        holdout_predictors_, holdout_responses_, model_->final_state()));
    errors_->row(i) = all_errors;
  }
}

HoldoutErrorSampler StateSpaceRegressionModelManager::CreateHoldoutSampler(
    SEXP r_bsts_object,
    int cutpoint,
    Matrix *prediction_error_output) {
  RListIoManager io_manager;
  Ptr<StateSpaceRegressionModel> model =
      static_cast<StateSpaceRegressionModel *>(CreateModel(
          R_NilValue,
          getListElement(r_bsts_object, "state.specification"),
          getListElement(r_bsts_object, "prior"),
          getListElement(r_bsts_object, "model.options"),
          nullptr,
          &io_manager));
  AddDataFromBstsObject(r_bsts_object);

  std::vector<Ptr<StateSpace::MultiplexedRegressionData>> data = model->dat();
  model->clear_data();
  for (int i = 0; i <= cutpoint; ++i) {
    model->add_multiplexed_data(data[i]);
  }
  int holdout_sample_size = 0;
  for (int i = cutpoint + 1; i < data.size(); ++i) {
    holdout_sample_size += data[i]->total_sample_size();
  }
  Matrix holdout_predictors(holdout_sample_size,
                            model->observation_model()->xdim());
  Vector holdout_response(holdout_sample_size);
  int index = 0;
  for (int i = cutpoint + 1; i < data.size(); ++i) {
    for (int j = 0; j < data[i]->total_sample_size(); ++j) {
      holdout_predictors.row(index) = data[i]->regression_data(j).x();
      holdout_response[index] = data[i]->regression_data(j).y();
      ++index;
    }
  }
  return HoldoutErrorSampler(new ErrorSampler(
      model, holdout_response, holdout_predictors,
      Rf_asInteger(getListElement(r_bsts_object, "niter")),
      prediction_error_output));
}

}  // namespace bsts
}  // namespace BOOM

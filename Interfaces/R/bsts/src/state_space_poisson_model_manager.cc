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

#include "state_space_poisson_model_manager.h"
#include "utils.h"

#include "r_interface/prior_specification.hpp"

#include "cpputil/math_utils.hpp"

#include "Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/MvnModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePoissonPosteriorSampler.hpp"

namespace BOOM {
namespace bsts {

namespace {
  typedef StateSpacePoissonModelManager SSPMM;
  void zero_missing_values(
      Vector *counts,
      Vector *exposure,
      const std::vector<bool> &observed) {
    for (size_t i = 0; i < counts->size(); ++i) {
      if (!observed[i]) {
        (*counts)[i] = (*exposure)[i] = 0;
      }
    }
  }
}   // namespace

SSPMM::StateSpacePoissonModelManager()
    : predictor_dimension_(-1) {}

StateSpacePoissonModel * SSPMM::CreateObservationModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) {
  if (!Rf_isNull(r_data_list)) {
    // If we were passed data from R then build the model using the
    // data that was passed.
    bool regression = !Rf_isNull(getListElement(r_data_list, "predictors"));
    Vector counts(ToBoomVector(getListElement(r_data_list, "response")));
    Vector exposure(ToBoomVector(getListElement(r_data_list, "exposure")));
    // If there are no predictors then make an intercept.
    Matrix predictors =
        regression ?
        ToBoomMatrix(getListElement(r_data_list, "predictors")) :
        Matrix(counts.size(), 1.0);
    std::vector<bool> response_is_observed(ToVectorBool(getListElement(
        r_data_list, "response.is.observed")));
    zero_missing_values(&counts, &exposure, response_is_observed);

    UnpackTimestampInfo(r_data_list);
    if (TimestampsAreTrivial()) {
      model_.reset(
          new StateSpacePoissonModel(
              counts,
              exposure,
              predictors,
              response_is_observed));
    } else {
      model_.reset(new StateSpacePoissonModel(predictors.ncol()));
      std::vector<Ptr<StateSpace::AugmentedPoissonRegressionData>> data;
      data.reserve(NumberOfTimePoints());
      for (int i = 0; i < NumberOfTimePoints(); ++i) {
        data.push_back(new StateSpace::AugmentedPoissonRegressionData);
      }
      for (int i = 0; i < counts.size(); ++i) {
        NEW(PoissonRegressionData, data_point)(counts[i],
                                               predictors.row(i),
                                               exposure[i]);
        if (!response_is_observed[i]) {
          data_point->set_missing_status(Data::completely_missing);
        }
        data[TimestampMapping(i)]->add_data(data_point);
      }
      for (int i = 0; i < NumberOfTimePoints(); ++i) {
        if (data[i]->observed_sample_size() == 0) {
          data[i]->set_missing_status(Data::completely_missing);
        }
        model_->add_data(data[i]);
      }
    }
    // With the Gaussian models we have two separate classes for the
    // regression and non-regression cases.  For non-Gaussian models
    // we have a single class with a regression bit that can be turned
    // off.
    model_->set_regression_flag(regression);
  } else {
    // If no data was passed from R then build the model from its
    // default constructor.  We need to know the dimension of the
    // predictors.
    if (predictor_dimension_ < 0) {
      report_error("If r_data_list is NULL then you must call "
                   "SetPredictorDimension before calling CreateModel.");
    }
    model_.reset(new StateSpacePoissonModel(predictor_dimension_));
  }

  Ptr<PoissonRegressionSpikeSlabSampler> observation_model_sampler;
  if (!Rf_isNull(r_prior)
      && Rf_inherits(r_prior, "SpikeSlabPriorBase")) {
    // If r_prior is NULL it could either mean that there are no
    // predictors, or that an existing model is being reinstantiated.
    RInterface::SpikeSlabGlmPrior prior_spec(r_prior);
    observation_model_sampler = new PoissonRegressionSpikeSlabSampler(
        model_->observation_model(),
        prior_spec.slab(),
        prior_spec.spike());
    DropUnforcedCoefficients(
        model_->observation_model(),
        prior_spec.spike()->prior_inclusion_probabilities());
    // Restrict number of model selection sweeps if max_flips was set.
    int max_flips = prior_spec.max_flips();
    if (max_flips > 0) {
      observation_model_sampler->limit_model_selection(max_flips);
    }
    // Make the io_manager aware of the model parameters.
    io_manager->add_list_element(
        new GlmCoefsListElement(model_->observation_model()->coef_prm(),
                                "coefficients"));
  } else {
    // In the non-regression (or no sampler necessary) case make a
    // spike and slab prior that never includes anything.
    observation_model_sampler = new PoissonRegressionSpikeSlabSampler(
        model_->observation_model(),
        new MvnModel(1),
        new VariableSelectionPrior(1, 0.0));
  }
  // Both the observation_model and the actual model_ need to have
  // their posterior samplers set.
  model_->observation_model()->set_method(observation_model_sampler);
  Ptr<StateSpacePoissonPosteriorSampler> sampler(
      new StateSpacePoissonPosteriorSampler(
          model_.get(),
          observation_model_sampler));

  if (!Rf_isNull(r_options)
      && !Rf_asLogical(getListElement(r_options, "enable.threads"))) {
    sampler->disable_threads();
  }

  model_->set_method(sampler);
  return model_.get();
}

void SSPMM::AddDataFromBstsObject(SEXP r_bsts_object) {
  SEXP r_counts = getListElement(r_bsts_object, "original.series");
  Vector counts = ToBoomVector(r_counts);
  AddData(counts,
          ToBoomVector(getListElement(r_bsts_object, "exposure")),
          ExtractPredictors(r_bsts_object, "predictors", counts.size()),
          IsObserved(r_counts));
}

void SSPMM::AddDataFromList(SEXP r_data_list) {
  SEXP r_counts = getListElement(r_data_list, "response");
  Vector counts = ToBoomVector(r_counts);
  AddData(counts,
          ToBoomVector(getListElement(r_data_list, "exposure")),
          ExtractPredictors(r_data_list, "predictors", counts.size()),
          ToVectorBool(getListElement(r_data_list, "response.is.observed")));
}

int SSPMM::UnpackForecastData(SEXP r_prediction_data) {
  UnpackForecastTimestamps(r_prediction_data);
  forecast_exposure_ = ToBoomVector(getListElement(
      r_prediction_data, "exposure"));
  int n = forecast_exposure_.size();
  forecast_predictors_ = ExtractPredictors(r_prediction_data, "predictors", n);
  return n;
}

Vector SSPMM::SimulateForecast(const Vector &final_state) {
  if (ForecastTimestamps().empty()) {
    return model_->simulate_forecast(
        rng(), forecast_predictors_, forecast_exposure_, final_state);
  } else {
    return model_->simulate_multiplex_forecast(
        rng(), forecast_predictors_, forecast_exposure_, final_state,
        ForecastTimestamps());
  }
}

void SSPMM::SetPredictorDimension(int xdim) {
  predictor_dimension_ = xdim;
}

void SSPMM::AddData(const Vector &counts,
                    const Vector &exposure,
                    const Matrix &predictors,
                    const std::vector<bool> &is_observed) {
  for (int i = 0; i < counts.size(); ++i) {
    bool missing = (!is_observed.empty() && !is_observed[i]);
    Ptr<StateSpace::AugmentedPoissonRegressionData> data_point(
        new StateSpace::AugmentedPoissonRegressionData(
            missing ? 0 : counts[i],
            missing ? 0 : exposure[i],
            predictors.row(i)));
    if (missing) {
      data_point->set_missing_status(Data::missing_status::completely_missing);
    }
    model_->add_data(data_point);
  }
}

}  // namespace bsts
}  // namespace BOOM

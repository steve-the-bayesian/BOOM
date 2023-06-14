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

#include "state_space_student_model_manager.h"
#include "utils.h"

#include "r_interface/prior_specification.hpp"
#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpaceStudentPosteriorSampler.hpp"

namespace BOOM {
  namespace bsts {

    namespace {
      typedef StateSpaceStudentModelManager SSSMM;
    }

    SSSMM::StateSpaceStudentModelManager()
        : predictor_dimension_(-1) {}

    StateSpaceStudentRegressionModel * SSSMM::CreateBareModel(
        SEXP r_data_list,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      Matrix predictors;
      Vector response;
      if (!Rf_isNull(r_data_list)) {
        std::vector<bool> response_is_observed;
        // If we were passed data from R then use it to build the model.
        SEXP r_predictors = getListElement(r_data_list, "predictors");
        if (Rf_inherits(r_data_list, "bsts")) {
          SEXP r_response = getListElement(r_data_list, "original.series");
          response = ToBoomVector(r_response);
          response_is_observed = IsObserved(r_response);
        } else {
          response = ToBoomVector(getListElement(r_data_list, "response"));
          response_is_observed = ToVectorBool(getListElement(
              r_data_list, "response.is.observed"));
        }
        bool regression = !Rf_isNull(r_predictors);

        // If no predictors were passed from R, create an intercept.
        predictors = regression ? ToBoomMatrix(r_predictors) :
            Matrix(response.size(), 1, 1.0);
        UnpackTimestampInfo(r_data_list);

        if (TimestampsAreTrivial()) {
          model_.reset(new StateSpaceStudentRegressionModel(
              response, predictors, response_is_observed));
        } else {
          // Nontrivial timestamps.
          int xdim = predictors.ncol();
          model_.reset(new StateSpaceStudentRegressionModel(xdim));
          std::vector<Ptr<StateSpace::AugmentedStudentRegressionData>> data;
          data.reserve(NumberOfTimePoints());
          for (int i = 0; i < NumberOfTimePoints(); ++i) {
            data.push_back(new StateSpace::AugmentedStudentRegressionData);
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
            model_->add_data(data[i]);
          }
        }
        model_->set_regression_flag(regression);
      } else {
        // If no data was passed from R then build the model from its
        // default constructor.  We need to know the dimension of the
        // predictors.
        if (predictor_dimension_ < 0) {
          report_error("If r_data_list is NULL then you must call "
                       "SetPredictorDimension before creating a model.");
        } else if (predictor_dimension_ == 0) {
          // If there are no predictors, make room for a dummy intercept.
          model_.reset(new StateSpaceStudentRegressionModel(1));
        } else {
          model_.reset(new StateSpaceStudentRegressionModel(predictor_dimension_));
        }
      }

      // A NULL r_prior signals that no posterior sampler is needed.  This
      // differs from the logit and Poisson cases, where a NULL prior might
      // signal the absence of predictors, because the T model still needs a
      // prior for the residual "variance" and tail thickness parameters.
      if (!Rf_isNull(r_prior)) {
        TRegressionModel *regression = model_->observation_model();
        BOOM::RInterface::StudentRegressionConjugateSpikeSlabPrior prior_spec(
            r_prior, regression->Sigsq_prm());
        Ptr<TRegressionSpikeSlabSampler> observation_model_sampler(
            new TRegressionSpikeSlabSampler(
                regression,
                prior_spec.slab(),
                prior_spec.spike(),
                prior_spec.siginv_prior(),
                prior_spec.degrees_of_freedom_prior()));
        DropUnforcedCoefficients(regression,
                                 prior_spec.prior_inclusion_probabilities());
        // Restrict number of attempted flips and the domain of the
        // residual "standard deviation" if these have been set.
        observation_model_sampler->set_sigma_upper_limit(
            prior_spec.sigma_upper_limit());
        int max_flips = prior_spec.max_flips();
        if (max_flips > 0) {
          observation_model_sampler->limit_model_selection(max_flips);
        }
        // Both the observation_model and the actual model_ need to have
        // their posterior samplers set.
        regression->set_method(observation_model_sampler);
        Ptr<StateSpaceStudentPosteriorSampler> sampler(
            new StateSpaceStudentPosteriorSampler(
                model_.get(),
                observation_model_sampler));
        if (!Rf_isNull(r_options)
            && !Rf_asLogical(getListElement(r_options, "enable.threads"))) {
          sampler->disable_threads();
        }
        model_->set_method(sampler);
      }

      // Make the io_manager aware of all the model parameters.
      if (model_->observation_model()->xdim() > 1) {
        io_manager->add_list_element(
            new GlmCoefsListElement(
                model_->observation_model()->coef_prm(),
                "coefficients"));
      }
      io_manager->add_list_element(
          new StandardDeviationListElement(
              model_->observation_model()->Sigsq_prm(),
              "sigma.obs"));
      io_manager->add_list_element(
          new UnivariateListElement(
              model_->observation_model()->Nu_prm(),
              "observation.df"));

      return model_.get();
    }

    HoldoutErrorSampler SSSMM::CreateHoldoutSampler(
        SEXP r_bsts_object,
        int cutpoint,
        bool standardize,
        Matrix *prediction_error_output) {
      RListIoManager io_manager;
      Ptr<StateSpaceStudentRegressionModel> model =
          static_cast<StateSpaceStudentRegressionModel *>(CreateModel(
              R_NilValue,
              getListElement(r_bsts_object, "state.specification"),
              getListElement(r_bsts_object, "prior"),
              getListElement(r_bsts_object, "model.options"),
              &io_manager));
      AddDataFromBstsObject(r_bsts_object);

      std::vector<Ptr<StateSpace::AugmentedStudentRegressionData>> data =
          model->dat();
      model->clear_data();
      for (int i = 0; i <= cutpoint; ++i) {
        model->add_data(data[i]);
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
      return HoldoutErrorSampler(new StateSpaceStudentHoldoutErrorSampler(
          model, holdout_response, holdout_predictors,
          Rf_asInteger(getListElement(r_bsts_object, "niter")),
          standardize,
          prediction_error_output));
    }

    void SSSMM::AddDataFromBstsObject(SEXP r_bsts_object) {
      SEXP r_response = getListElement(r_bsts_object, "original.series");
      Vector response = ToBoomVector(r_response);
      AddData(response,
              ExtractPredictors(r_bsts_object, "predictors", response.size()),
              IsObserved(r_response));
    }

    void SSSMM::AddDataFromList(SEXP r_data_list) {
      Vector response = ToBoomVector(getListElement(r_data_list, "response"));
      AddData(response,
              ExtractPredictors(r_data_list, "predictors", response.size()),
              ToVectorBool(getListElement(r_data_list, "response.is.observed")));
    }

    int SSSMM::UnpackForecastData(SEXP r_prediction_data) {
      UnpackForecastTimestamps(r_prediction_data);
      SEXP r_predictors = getListElement(r_prediction_data, "predictors");
      if (!Rf_isNull(r_predictors)) {
        forecast_predictors_ = ToBoomMatrix(r_predictors);
      } else {
        int horizon = Rf_asInteger(getListElement(
            r_prediction_data, "horizon"));
        forecast_predictors_ = Matrix(horizon, 1, 1.0);
      }
      return forecast_predictors_.nrow();
    }

    Vector SSSMM::SimulateForecast(const Vector &final_state) {
      if (ForecastTimestamps().empty()) {
        return model_->simulate_forecast(rng(), forecast_predictors_, final_state);
      } else {
        return model_->simulate_multiplex_forecast(
            rng(), forecast_predictors_, final_state, ForecastTimestamps());
      }
    }

    void SSSMM::AddData(const Vector &response,
                        const Matrix &predictors,
                        const std::vector<bool> &response_is_observed) {
      int sample_size = response.size();
      for (int i = 0; i < sample_size; ++i) {
        Ptr<StateSpace::AugmentedStudentRegressionData> data_point(
            new StateSpace::AugmentedStudentRegressionData(
                response[i],
                predictors.row(i)));
        if (!response_is_observed.empty() && !response_is_observed[i]) {
          data_point->set_missing_status(Data::missing_status::completely_missing);
        }
        model_->add_data(data_point);
      }
    }

    void SSSMM::SetPredictorDimension(int xdim) {
      predictor_dimension_ = xdim;
    }

  }  // namespace bsts
}  // namespace BOOM

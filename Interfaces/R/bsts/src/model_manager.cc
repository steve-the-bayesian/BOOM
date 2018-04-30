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

#include <string>

#include "model_manager.h"
#include "state_space_gaussian_model_manager.h"
#include "state_space_logit_model_manager.h"
#include "state_space_poisson_model_manager.h"
#include "state_space_regression_model_manager.h"
#include "state_space_student_model_manager.h"
#include "utils.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/create_state_model.hpp"
#include "r_interface/list_io.hpp"

#include "Models/StateSpace/Filters/KalmanTools.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"

#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace bsts {

    // The model manager will be thread safe as long as it is created from the
    // home thread.
    ModelManager::ModelManager()
        : rng_(seed_rng(GlobalRng::rng)),
          timestamps_are_trivial_(true),
          number_of_time_points_(-1) {}

    ModelManager * ModelManager::Create(SEXP r_bsts_object) {  // NOLINT
      std::string family = ToString(getListElement(
          r_bsts_object, "family"));
      bool regression = !Rf_isNull(getListElement(
          r_bsts_object, "predictors"));
      int xdim = 0;
      if (regression) {
        xdim = Rf_ncols(getListElement(r_bsts_object, "predictors"));
      }
      return ModelManager::Create(family, xdim);
    }

    ModelManager *ModelManager::Create(const std::string &family_name,  
                                       int xdim) {
      ModelManager *ans = nullptr;
      if (family_name == "gaussian") {
        if (xdim > 0) {
          StateSpaceRegressionModelManager *manager =
              new StateSpaceRegressionModelManager;
          manager->SetPredictorDimension(xdim);
          ans = manager;
        } else {
          ans = new StateSpaceModelManager;
        }
      } else if (family_name == "logit") {
        StateSpaceLogitModelManager *manager =
            new StateSpaceLogitModelManager;
        manager->SetPredictorDimension(xdim);
        ans = manager;
      } else if (family_name == "poisson") {
        StateSpacePoissonModelManager *manager =
            new StateSpacePoissonModelManager;
        manager->SetPredictorDimension(xdim);
        ans = manager;
      } else if (family_name == "student") {
        StateSpaceStudentModelManager *manager
            = new StateSpaceStudentModelManager;
        manager->SetPredictorDimension(xdim);
        ans = manager;
      } else {
        std::ostringstream err;
        err << "Unrecognized family name: " << family_name
            << " in ModelManager::Create.";
        report_error(err.str());
        return nullptr;
      }
      return ans;
    }

    class FullStateCallback : public MatrixIoCallback {
     public:
      explicit FullStateCallback(StateSpaceModelBase *model) : model_(model) {}
      int nrow() const override {return model_->state_dimension();}
      int ncol() const override {return model_->time_dimension();}
      Matrix get_matrix() const override {return model_->state();}
     private:
      StateSpaceModelBase *model_;
    };
    
    ScalarStateSpaceModelBase * ModelManager::CreateModel(  // NOLINT
        SEXP r_data_list,
        SEXP r_state_specification,
        SEXP r_prior,
        SEXP r_options,
        Vector *final_state,
        RListIoManager *io_manager) {
      ScalarStateSpaceModelBase *model = CreateObservationModel(
          r_data_list,
          r_prior,
          r_options,
          io_manager);
      RInterface::StateModelFactory state_model_factory(io_manager, model);
      state_model_factory.AddState(r_state_specification);
      state_model_factory.SaveFinalState(final_state);

      // The predict method does not set BstsOptions, so allow NULL here to
      // signal that options have not been set.
      if (!Rf_isNull(r_options)) {
        bool save_state_contribution = Rf_asLogical(getListElement(
            r_options, "save.state.contributions"));
        if (save_state_contribution) {
          io_manager->add_list_element(
              new NativeMatrixListElement(
                  new GeneralStateContributionCallback(model),
                  "state.contributions",
                  nullptr));
        }

        bool save_prediction_errors = Rf_asLogical(getListElement(
            r_options, "save.prediction.errors"));
        if (save_prediction_errors) {
          // The final nullptr argument is because we will not be streaming
          // prediction errors in future calculations.  They are for reporting
          // only.  As usual, the rows of the matrix correspond to MCMC
          // iterations, so the columns represent time.
          io_manager->add_list_element(
              new BOOM::NativeVectorListElement(
                  new PredictionErrorCallback(model),
                  "one.step.prediction.errors",
                  nullptr));
        }

        bool save_full_state = Rf_asLogical(getListElement(
            r_options, "save.full.state"));
        if (save_full_state) {
          io_manager->add_list_element(
              new NativeMatrixListElement(
                  new FullStateCallback(model), "full.state", nullptr));
        }
      }
      return model;
    }

    Matrix ModelManager::Forecast(SEXP r_bsts_object,
                                  SEXP r_prediction_data,
                                  SEXP r_burn,
                                  SEXP r_observed_data) {
      RListIoManager io_manager;
      Vector final_state;
      SEXP r_state_specfication = getListElement(
          r_bsts_object, "state.specification");
      ScalarStateSpaceModelBase *model = CreateModel(
          R_NilValue,
          r_state_specfication,
          R_NilValue,
          R_NilValue,
          &final_state,
          &io_manager);
      bool refilter;
      if (Rf_isNull(r_observed_data)) {
        AddDataFromBstsObject(r_bsts_object);
        refilter = false;
      } else {
        AddDataFromList(r_observed_data);
        refilter = true;
      }
      int niter = Rf_asInteger(getListElement(r_bsts_object, "niter"));
      int burn = std::max<int>(0, Rf_asInteger(r_burn));
      io_manager.prepare_to_stream(r_bsts_object);
      io_manager.advance(burn);
      int iterations_after_burnin = niter - burn;

      if (Rf_isNull(r_prediction_data)) {
        report_error("Forecast called with NULL prediction data.");
      }
      int forecast_horizon = UnpackForecastData(r_prediction_data);
      UnpackDynamicRegressionForecastData(
          model, r_state_specfication, r_prediction_data);

      Matrix ans(iterations_after_burnin, forecast_horizon);
      for (int i = 0; i < iterations_after_burnin; ++i) {
        io_manager.stream();
        if (refilter) {
          model->light_kalman_filter();
          ScalarKalmanStorage storage = model->final_kalman_storage();
          Vector state_mean = storage.a;
          SpdMatrix state_variance = storage.P;
          make_contemporaneous(
              state_mean,
              state_variance,
              storage.F,
              storage.v,
              model->observation_matrix(model->time_dimension()).dense());
          final_state = rmvn(state_mean, state_variance);
        }
        ans.row(i) = SimulateForecast(final_state);
      }
      return ans;
    }

    void ModelManager::UnpackDynamicRegressionForecastData(
        ScalarStateSpaceModelBase *model,
        SEXP r_state_specification,
        SEXP r_prediction_data) {
      if (Rf_length(r_state_specification) < model->nstate()) {
        std::ostringstream err;
        err << "The number of state components in the model: ("
            << model->nstate() << ") does not match the size of "
            << "the state specification: ("
            << Rf_length(r_state_specification)
            << ") in UnpackDynamicRegressionForecastData.";
        report_error(err.str());
      }
      for (int i = 0; i < model->nstate(); ++i) {
        SEXP spec = VECTOR_ELT(r_state_specification, i);
        if (Rf_inherits(spec, "DynamicRegression")) {
          Matrix predictors = ToBoomMatrix(getListElement(
              r_prediction_data, "dynamic.regression.predictors"));
          Ptr<StateModel> state_model = model->state_model(i);
          state_model.dcast<DynamicRegressionStateModel>()->add_forecast_data(
              predictors);
        }
      }
    }

    void ModelManager::UnpackTimestampInfo(SEXP r_data_list) {
      SEXP r_timestamp_info = getListElement(r_data_list, "timestamp.info");
      timestamps_are_trivial_ = Rf_asLogical(getListElement(
          r_timestamp_info, "timestamps.are.trivial"));
      number_of_time_points_ = Rf_asInteger(getListElement(
          r_timestamp_info, "number.of.time.points"));
      if (!timestamps_are_trivial_) {
        timestamp_mapping_ = ToIntVector(getListElement(
            r_timestamp_info, "timestamp.mapping"));
      }
    }

    void ModelManager::UnpackForecastTimestamps(SEXP r_prediction_data) {
      SEXP r_forecast_timestamps = getListElement(
          r_prediction_data, "timestamps");
      if (!Rf_isNull(r_forecast_timestamps)) {
        forecast_timestamps_ = ToIntVector(getListElement(
            r_forecast_timestamps, "timestamp.mapping"));
        for (int i = 1; i < forecast_timestamps_.size(); ++i) {
          if (forecast_timestamps_[i] < forecast_timestamps_[i - 1]) {
            report_error("Time stamps for multiplex predictions must be "
                         "in increasing order.");
          }
        }
      }
    }

  }  // namespace bsts
}  // namespace BOOM

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
#include "multivariate_gaussian_model_manager.h"

#include "utils.h"
#include "create_state_model.h"
#include "create_shared_state_model.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"

#include "Models/StateSpace/Filters/KalmanTools.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionArStateModel.hpp"

#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace bsts {

    //==========================================================================

    // The model manager will be thread safe as long as it is created from the
    // home thread.
    ModelManager::ModelManager()
        : rng_(seed_rng(GlobalRng::rng)) {}

    ScalarModelManager * ScalarModelManager::Create(SEXP r_bsts_object) {
      std::string family = ToString(getListElement(r_bsts_object, "family"));
      bool regression = !Rf_isNull(getListElement(r_bsts_object, "predictors"));
      int xdim = 0;
      if (regression) {
        xdim = Rf_ncols(getListElement(r_bsts_object, "predictors"));
      }
      return ScalarModelManager::Create(family, xdim);
    }

    ScalarModelManager *ScalarModelManager::Create(
        const std::string &family_name, int xdim) {
      if (family_name == "gaussian") {
        if (xdim > 0) {
          StateSpaceRegressionModelManager *manager =
              new StateSpaceRegressionModelManager;
          manager->SetPredictorDimension(xdim);
          return manager;
        } else {
          return new StateSpaceModelManager;
        }
      } else if (family_name == "logit") {
        StateSpaceLogitModelManager *manager =
            new StateSpaceLogitModelManager;
        manager->SetPredictorDimension(xdim);
        return manager;
      } else if (family_name == "poisson") {
        StateSpacePoissonModelManager *manager =
            new StateSpacePoissonModelManager;
        manager->SetPredictorDimension(xdim);
        return manager;
      } else if (family_name == "student") {
        StateSpaceStudentModelManager *manager
            = new StateSpaceStudentModelManager;
        manager->SetPredictorDimension(xdim);
        return manager;
      } else {
        std::ostringstream err;
        err << "Unrecognized family name: " << family_name
            << " in ModelManager::Create.";
        report_error(err.str());
      }
      return nullptr;
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

    ScalarStateSpaceModelBase * ScalarModelManager::CreateModel(
        SEXP r_data_list,
        SEXP r_state_specification,
        SEXP r_prior,
        SEXP r_options,
        RListIoManager *io_manager) {
      ScalarStateSpaceModelBase *model = CreateBareModel(
          r_data_list,
          r_prior,
          r_options,
          io_manager);
      StateModelFactory state_model_factory(io_manager);
      state_model_factory.AddState(model, r_state_specification, "");
      state_model_factory.SaveFinalState(model, &final_state());

      // The predict method does not set BstsOptions, so allow NULL here to
      // signal that options have not been set.
      if (!Rf_isNull(r_options)) {
        bool save_state_contribution = Rf_asLogical(getListElement(
            r_options, "save.state.contributions"));
        if (save_state_contribution) {
          io_manager->add_list_element(
              new NativeMatrixListElement(
                  new ScalarStateContributionCallback(model),
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

    // Primary implementation of predict.bsts.  Child classes will carry out
    // some of the details, but most of the prediction logic is here.
    //
    // Args:
    //   r_bsts_object:  The bsts model object for which a prediction is deisred.
    //   r_prediction_data: An R list containing any additional data needed to
    //     make the prediction.  For simple state space models this is just an
    //     integer giving the time horizon over which to predict.  For models
    //     containing a regression component it contains the future values of
    //     the X's.  For binomial (or Poisson) models it contains a sequence of
    //     future trial counts (or exposures).
    //   r_burn: An integer giving the number of burn-in iterations to discard.
    //     Negative numbers will be treated as zero.  Numbers greater than the
    //     number of MCMC iterations will raise an error.
    //   r_observed_data: An R list containing the observed data on which to
    //     base the prediction.  In typical cases this should be R_NilValue (R's
    //     NULL) signaling that the observed data should be taken from
    //     r_bsts_object.  However, if additional data have been observed since
    //     the model was trained, or if the model is being used to predict data
    //     that were part of the training set, or some other application other
    //     than making predictions starting from one time period after the
    //     training data ends, then one can use this argument to pass the
    //     "training data" on which the predictions should be based.  If this
    //     argument is used, then the Kalman filter will be run over the
    //     supplied data, which is expensive.  If this argument is left as
    //     R_NilValue (NULL) then the "final.state" element of r_bsts_object
    //     will be used as the basis for future predictions, which is a
    //     computational savings over filtering from scratch.
    Matrix ScalarModelManager::Forecast(SEXP r_bsts_object,
                                        SEXP r_prediction_data,
                                        SEXP r_burn,
                                        SEXP r_observed_data) {
      RListIoManager io_manager;
      SEXP r_state_specfication = getListElement(
          r_bsts_object, "state.specification");
      ScalarStateSpaceModelBase *model = CreateModel(
          R_NilValue,
          r_state_specfication,
          R_NilValue,
          R_NilValue,
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
      if (burn < 0) burn = 0;
      io_manager.prepare_to_stream(r_bsts_object);
      io_manager.advance(burn);
      int iterations_after_burnin = niter - burn;

      if (Rf_isNull(r_prediction_data)) {
        report_error("Forecast called with NULL prediction data.");
      }
      int forecast_horizon = UnpackForecastData(r_prediction_data);
      UnpackDynamicRegressionForecastData(r_prediction_data, model);
      int max_time = model->time_dimension() + forecast_horizon;
      for (int s = 0; s < model->number_of_state_models(); ++s) {
        model->state_model(s)->observe_time_dimension(max_time);
      }
      Matrix ans(iterations_after_burnin, forecast_horizon);
      for (int i = 0; i < iterations_after_burnin; ++i) {
        io_manager.stream();
        Vector final_state = this->final_state();
        if (refilter) {
          model->kalman_filter();
          const Kalman::ScalarMarginalDistribution &marg(
              model->get_filter().back());
          Vector state_mean = marg.state_mean();
          SpdMatrix state_variance = marg.state_variance();
          make_contemporaneous(
              state_mean,
              state_variance,
              marg.prediction_variance(),
              marg.prediction_error(),
              model->observation_matrix(model->time_dimension()).dense());
          final_state = rmvn(state_mean, state_variance);
        }
        ans.row(i) = SimulateForecast(final_state);
      }
      return ans;
    }

    void ScalarModelManager::UnpackDynamicRegressionForecastData(
        SEXP r_prediction_data, ScalarStateSpaceModelBase *model) {
      SEXP r_dynamic_regression_predictors = getListElement(
          r_prediction_data, "dynamic.regression.predictors");
      if (Rf_isNull(r_dynamic_regression_predictors)) {
        return;
      }
      for (int s = 0; s < model->number_of_state_models(); ++s) {
        DynamicRegressionStateModel *dreg =
            dynamic_cast<DynamicRegressionStateModel *>(
                model->state_model(s));
        if (dreg) {
          dreg->add_forecast_data(ToBoomMatrix(
              r_dynamic_regression_predictors));
          return;
        }

        DynamicRegressionArStateModel *dregar =
            dynamic_cast<DynamicRegressionArStateModel *>(
                model->state_model(s));
        if (dregar) {
          dregar->add_forecast_data(ToBoomMatrix(
              r_dynamic_regression_predictors));
          return;
        }
      }
    }

    //=========================================================================
    MultivariateModelManagerBase * MultivariateModelManagerBase::Create(
        SEXP r_mbsts_object) {
      std::string family = ToString(getListElement(r_mbsts_object, "family"));
      int nseries = Rf_ncols(getListElement(
          r_mbsts_object, "original.series", true));
      bool regression = !Rf_isNull(getListElement(
          r_mbsts_object, "predictors", true));
      int xdim = 0;
      if (regression) {
        xdim = Rf_ncols(getListElement(r_mbsts_object, "predictors"));
      }
      return MultivariateModelManagerBase::Create(family, nseries, xdim);
    }

    //--------------------------------------------------------------------------
    MultivariateModelManagerBase * MultivariateModelManagerBase::Create(
        const std::string &family, int nseries, int xdim) {

      if (family == "gaussian") {
        MultivariateGaussianModelManager *manager =
            new MultivariateGaussianModelManager(nseries, xdim);
        return manager;
      } else {
        report_error("For now, only Gaussian families are supported in the "
                     "multivariate case.");
      }
      return nullptr;
    }

    //--------------------------------------------------------------------------

  }  // namespace bsts
}  // namespace BOOM

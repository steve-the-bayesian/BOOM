#ifndef BSTS_DYNAMIC_INTERCEPT_MODEL_MANAGER_H_
#define BSTS_DYNAMIC_INTERCEPT_MODEL_MANAGER_H_

// Copyright 2018 Steven L. Scott. All Rights Reserved.
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

#include "model_manager.h"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"

namespace BOOM {
  namespace bsts {

    class DynamicInterceptModelManager
        : public ModelManager {
   public:
      // Args:
      //   xdim:  The number of predictor variables in the regression model.
      DynamicInterceptModelManager(int xdim);

      // Create a DynamicInterceptModelManager from a data list passed by R.  
      static DynamicInterceptModelManager *Create(SEXP r_data_list);

      // Create a new DynamicInterceptRegressionModel.
      // Args:
      //   r_data_list: An R list containing 'predictors' (a matrix) and
      //     'response' (a vector).
      //   r_state_specification: An R list of state specification objects used
      //     to define the dynamic intercept term in the regression model.
      //   r_prior: An R object of class SpikeSlabPrior defining the prior
      //     distribution for the regression component of the model.
      //   r_options: A list containing 'large.sample.threshold.factor', a
      //     non-negative scalar that determines the cutoff between dense and
      //     sparse linear algebra in the Kalman filter updates for this model.
      //   io_manager: The io_manager responsible for writing MCMC output to an
      //     R object, or streaming it from an existing object.
      //
      // Returns:
      //  A pointer to the created model.  The pointer is owned by a Ptr
      //  in the model manager, and should be caught by a Ptr in the caller.
      //
      // Side Effects:
      //   The returned pointer is also held in a smart pointer in the model
      //   manager object that created it.
      DynamicInterceptRegressionModel * CreateModel(
          SEXP r_data_list,
          SEXP r_state_specification,
          SEXP r_prior,
          SEXP r_options,  
          RListIoManager *io_manager);

      // Args:
      //   r_dirm_object: The dynamic regression model object from R, to be used
      //     as the basis for the forecast.
      //   r_prediction_data:  A list containing elements
      //     - 'predictors': a matrix of predictor variables to be used for the forecast.
      //     - 'timestamps': A vector of time stamps, with length equal to the
      //          number of rows in 'predictors'.  This should be the same class
      //          as the 'timestamps' argument used to fit the model in
      //          'r_dirm_object'.
      //   r_burn: The number of initial MCMC draws in 'r_dirm_object' to
      //     discard as burn-in.
      //   r_observed_data: In most cases, the prediction takes place starting
      //     with the time period immediately following the last observation in
      //     the training data.  If so then r_observed_data should be
      //     R_NilValue, and the observed data will be taken from r_bsts_object.
      //     However, if more data have been added (or if some data should be
      //     omitted) from the training data, a new set of training data can be
      //     passed here.
      //
      // Returns:
      //   An R matrix, with rows corresponding to MCMC draws and columns to
      //   time, containing posterior predictive draws for the forecast.
      Matrix Forecast(SEXP r_dirm_object,
                      SEXP r_prediction_data,
                      SEXP r_burn,
                      SEXP r_observed_data); 

      // If the model contains any dynamic regression state components, the
      // model manager needs to know where they are located in the model, so
      // that the appropriate predictors can be assigned during forecasting.
      //
      // Args:
      //   positions: A vector with length matching the number of dynamic
      //     regression components in the model.  This will most often be 1 or
      //     0.  Each element is the index of a dynamic regression state
      //     component, so that a call to model->state_model(index) returns a
      //     dynamic regression state model.
      void SetDynamicRegressionStateComponentPositions(
          const std::vector<int> &positions) {
        dynamic_regression_state_positions_ = positions;
      }
      
      // TODO(steve): Move dynamic_regression_state_positions to the right model
      // manager for dynamic regression models.
      //
      // The locations (indices) in the vector of state models of any dynamic
      // regression state components.
      const std::vector<int> & dynamic_regression_state_positions() const {
        return dynamic_regression_state_positions_;
      }

   protected:
      int UnpackForecastData(SEXP r_prediction_data) override;

      Vector &final_state() {return final_state_;}
      
   private:
      void AddData(const Vector &response, const Matrix &predictors,
                   const Selector &response_is_observed);
      void AddDataFromList(SEXP r_data_list) override;
      void AddDataFromBstsObject(SEXP r_bsts_object) override;

      // Unpacks forecast data for the dynamic regression state component,
      // if one is present in the model.
      // Args:
      //   model:  The model to be forecast.
      //   r_state_specification: The R list of state specfication elements,
      //     used to determine the position of the dynamic regression component.
      //   r_prediction_data: A list.  If a dynamic regression component is
      //     present this list must contain an element named
      //     "dynamic.regression.predictors", which is an R matrix containing
      //     the forecast predictors for the dynamic regression component.
      void UnpackDynamicRegressionForecastData(
          DynamicInterceptRegressionModel *model,
          SEXP r_state_specification,
          SEXP r_prediction_data);

      Ptr<DynamicInterceptRegressionModel> model_;

      // The predictor matrix for forecasting.
      Matrix forecast_predictors_;

      Vector final_state_;

      // The index of each dynamic regression state component is stored here,
      // where 'index' refers to the state component's position in the list of
      // state models stored in the primary state space model.
      std::vector<int> dynamic_regression_state_positions_;
    };
    
  } // namespace bsts
}  // namespace BOOM 
#endif //  BSTS_DYNAMIC_INTERCEPT_MODEL_MANAGER_H_


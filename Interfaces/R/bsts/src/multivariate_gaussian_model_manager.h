#ifndef BOOM_RPACKAGES_BSTS_STATE_SPACE_MULTIVARIATE_GAUSSIAN_MODEL_MANAGER_H_
#define BOOM_RPACKAGES_BSTS_STATE_SPACE_MULTIVARIATE_GAUSSIAN_MODEL_MANAGER_H_
/*
  Copyright (C) 2019 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "model_manager.h"
#include "r_interface/boom_r_tools.hpp"

#include "LinAlg/Selector.hpp"
#include "Models/StateSpace/MultivariateStateSpaceRegressionModel.hpp"

namespace BOOM {
  namespace bsts {

    class MultivariateGaussianModelManager
        : public MultivariateModelManagerBase {
      
     public:
      // When you create the observation model add the state contributions to
      // the io_manager.

      // Args:
      //   ydim: The number of time series being modeled.
      //   xdim: The dimension of the predictor variables.  This can be zero,
      //     indicating that no predictors are present.  A negative value is a
      //     signal that the predictor dimension will be set later.
      MultivariateGaussianModelManager(int ydim, int xdim);

      // Handle model creation.  See comments in model_manager.h.
      MultivariateStateSpaceRegressionModel *CreateModel(
          SEXP r_data_list,
          SEXP r_shared_state_specification,
          SEXP r_series_state_specification,
          SEXP r_prior,
          SEXP r_options,
          RListIoManager *io_manager) override;

      void AddDataFromBstsObject(SEXP r_bsts_object) override;
      void AddDataFromList(SEXP r_data_list) override;
      int UnpackForecastData(SEXP r_prediction_data) override;

      // Forecast future values of the multivariate time series.
      //
      // Args:
      //   r_mbsts_object:  The model object created by 'mbsts.'
      //   r_prediction_data: An R list containing any additional data needed to
      //     make the prediction.  For simple state space models this is just an
      //     integer giving the time horizon over which to predict.  For models
      //     containing a regression component it contains the future values of
      //     the X's, along with 'series' and 'timestamps' for each X vector.
      //   r_burn: An integer giving the number of burn-in iterations to discard.
      //     Negative numbers will be treated as zero.  Numbers greater than the
      //     number of MCMC iterations will raise an error.
      Array Forecast(SEXP r_mbsts_object,
                     SEXP r_prediction_data,
                     SEXP r_burn) override;
      
     private:
      // Args:
      //   r_data_list:  A list that contains the following elements:
      //     - response:  A numeric vector.
      //     - predictors: A matrix.  The number of rows must equal the length of
      //       'response.'
      //     - timestamps: An R object of class TimestampInfo.  The timestamps
      //        are used to group individual responses into a response vector.
      //   r_prior: TBD.
      //   r_options:  Currently unused.
      //   io_manager: The input-output manager used to write to (and read from)
      //     the mbsts model object.
      //
      // Returns:
      //   The nearly fully formed model.  Data is assigned, as is the posterior
      //   sampler for the overall model, and model parameters are registered
      //   with the io_manager.  State is not assigned.
      MultivariateStateSpaceRegressionModel * CreateBareModel(
          SEXP r_data_list,
          SEXP r_prior,
          SEXP r_options,
          RListIoManager *io_manager) override;

      // TODO: How to handle the observation indicators.
      void AddData(const ConstVectorView &responses,
                   const Matrix &predictors,
                   const Factor &series);

      void BuildModelAndAssignData(SEXP r_data_list);
      void AssignSampler(SEXP r_prior, SEXP r_options);
      void ConfigureIo(RListIoManager *io_manager);
      void SaveFinalState(RListIoManager *io_manager);
      void SetModelOptions(SEXP r_options);
      
      Ptr<MultivariateStateSpaceRegressionModel> model_;
      int nseries_;
      int predictor_dimension_;
      TimestampInfo timestamp_info_;

      // A matrix of predictor variables for the forecast.  The first nseries_
      // rows are for the 1-step forecast.  The next nseries_ rows are for the
      // 2-step forecast, etc.  
      Matrix forecast_predictors_;

      std::vector<BOOM::Vector> series_specific_final_state_;
    };
    
  }  // namespace bsts
  
}  // namespace BOOM



#endif  // BOOM_RPACKAGES_BSTS_STATE_SPACE_MULTIVARIATE_GAUSSIAN_MODEL_MANAGER_H

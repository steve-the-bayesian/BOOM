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
      //   sampler for the overall model.  State is not assigned here.
      MultivariateStateSpaceRegressionModel * CreateObservationModel(
          SEXP r_data_list,
          SEXP r_prior,
          SEXP r_options,
          RListIoManager *io_manager) override;

      void AddDataFromBstsObject(SEXP r_bsts_object) override;
      void AddDataFromList(SEXP r_data_list) override;
      int UnpackForecastData(SEXP r_prediction_data) override;

      // Forecast future values of the multivariate time series.
      Array Forecast(SEXP r_mbsts_object,
                     SEXP r_prediction_data,
                     SEXP r_burn,
                     SEXP r_observed_data) override;
      
     private:
      // TODO: How to handle the observation indicators.
      void AddData(const Matrix &responses,
                   const Matrix &predictors,
                   const SelectorMatrix &observed);

      void BuildModelAndAssignData(SEXP r_data_list);
      void AssignSampler(SEXP r_prior);
      void ConfigureIo(RListIoManager *io_manager);
      
      Ptr<MultivariateStateSpaceRegressionModel> model_;
      int ydim_;
      int predictor_dimension_;

      TimestampInfo timestamp_info_;
      
      Matrix forecast_predictors_;
    };
    
  }  // namespace bsts
  
}  // namespace BOOM



#endif  // BOOM_RPACKAGES_BSTS_STATE_SPACE_MULTIVARIATE_GAUSSIAN_MODEL_MANAGER_H

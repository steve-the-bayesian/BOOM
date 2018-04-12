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

#ifndef BSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_
#define BSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_

#include "model_manager.h"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"

namespace BOOM {
namespace bsts {

// A base class that handles "CreateModel" for both the regression and
// non-regression flavors of Gaussian models.
class GaussianModelManagerBase : public ModelManager {
 public:
  ScalarStateSpaceModelBase * CreateModel(
      SEXP r_data_list,
      SEXP r_state_specification,
      SEXP r_prior,
      SEXP r_options,
      Vector *final_state,
      RListIoManager *io_manager) override;
};

// A holdout error sampler for a plain Gaussian state space model.
class StateSpaceModelPredictionErrorSampler
    : public HoldoutErrorSamplerImpl {
 public:
  // Args:
  //   model:  The model containing data up to a specified cutpoint.
  //   holdout_data:  Observed values after the cutpoint.
  //   niter: The desired number of draws (MCMC iterations) from the posterior
  //     distribution.
  //   errors:  A matrix that will hold the output of the simulation.
  StateSpaceModelPredictionErrorSampler(const Ptr<StateSpaceModel> &model,
                                        const Vector &holdout_data,
                                        int niter,
                                        Matrix *errors);
  void sample_holdout_prediction_errors() override;

 private:
  Ptr<StateSpaceModel> model_;
  Vector holdout_data_;
  int niter_;
  Matrix *errors_;
};

class StateSpaceModelManager
    : public GaussianModelManagerBase {
 public:
  // Creates the model_ object, assigns a PosteriorSamper, and
  // allocates space in the io_manager for the objects in the
  // observation model.
  // Args:
  //   r_data_list: Contains a numeric vector named 'response' and a
  //     logical vector 'response.is.observed.'
  //   r_prior:  An R object of class SdPrior.
  //   r_options:  Not used.
  //   io_manager:  The io_manager that will record the MCMC draws.
  StateSpaceModel * CreateObservationModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) override;

  HoldoutErrorSampler CreateHoldoutSampler(
      SEXP r_bsts_object,
      int cutpoint,
      Matrix *prediction_error_output) override;

  void AddDataFromBstsObject(SEXP r_bsts_object) override;
  void AddDataFromList(SEXP r_data_list) override;
  int UnpackForecastData(SEXP r_prediction_data) override;
  Vector SimulateForecast(const Vector &final_state) override;

 private:
  void AddData(const Vector &response,
               const std::vector<bool> &response_is_observed);

  Ptr<StateSpaceModel> model_;
  int forecast_horizon_;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_STATE_SPACE_GAUSSIAN_MODEL_MANAGER_H_

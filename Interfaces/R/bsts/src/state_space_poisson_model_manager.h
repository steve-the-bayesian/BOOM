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

#ifndef BSTS_SRC_STATE_SPACE_POISSON_MODEL_MANAGER_H_
#define BSTS_SRC_STATE_SPACE_POISSON_MODEL_MANAGER_H_

#include "model_manager.h"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpacePoissonModel.hpp"

namespace BOOM {
namespace bsts {

class StateSpacePoissonModelManager
    : public ScalarModelManager {
 public:
  StateSpacePoissonModelManager();

  StateSpacePoissonModel * CreateBareModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) override;

  HoldoutErrorSampler CreateHoldoutSampler(SEXP, int, bool, Matrix *) override {
    return HoldoutErrorSampler(new NullErrorSampler);
  }

  void AddDataFromBstsObject(SEXP r_bsts_object) override;
  void AddDataFromList(SEXP r_data_list) override;
  int UnpackForecastData(SEXP r_prediction_data) override;
  Vector SimulateForecast(const Vector &final_state) override;

  void SetPredictorDimension(int xdim);
  void AddData(const Vector &counts,
               const Vector &exposure,
               const Matrix &predictors,
               const std::vector<bool> &is_observed);

 private:
  Ptr<StateSpacePoissonModel> model_;
  int predictor_dimension_;

  Vector forecast_exposure_;
  Matrix forecast_predictors_;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_STATE_SPACE_POISSON_MODEL_MANAGER_H_

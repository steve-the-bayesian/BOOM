#ifndef ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_POISSON_MODEL_MANAGER_H_
#define ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_POISSON_MODEL_MANAGER_H_

#include "model_manager.h"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpacePoissonModel.hpp"

namespace BOOM {
namespace bsts {

class StateSpacePoissonModelManager
    : public ModelManager {
 public:
  StateSpacePoissonModelManager();

  StateSpacePoissonModel * CreateObservationModel(
      SEXP r_data_list,
      SEXP r_prior,
      SEXP r_options,
      RListIoManager *io_manager) override;

  HoldoutErrorSampler CreateHoldoutSampler(
      SEXP r_bsts_object, int cutpoint, Matrix *err) override {
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

#endif  // ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_POISSON_MODEL_MANAGER_H_

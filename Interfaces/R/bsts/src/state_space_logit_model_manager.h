#ifndef ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_LOGIT_MODEL_MANAGER_H_
#define ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_LOGIT_MODEL_MANAGER_H_

#include "model_manager.h"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/StateSpace/StateSpaceLogitModel.hpp"

namespace BOOM {
namespace bsts {

class StateSpaceLogitModelManager
    : public ModelManager {
 public:
  StateSpaceLogitModelManager();

  // Args:
  //   r_data_list: Must either be NULL, or contain 'response',
  //     'trials', and 'respnose.is.observed'.  If the model contains
  //     a regression component then it must contain 'predictors' as
  //     well.
  //   r_prior: Can be R_NilValue if the model has no regression
  //     component (or the model is not being created for MCMC).
  //     Otherwise this should be SpikeSlabGlmPrior.
  //   r_options: A list containing "clt.threshold" for use with the
  //     MCMC sampler.  Can be NULL.
  //   io_manager: The io_manager that will link the MCMC draws to the
  //     R list receiving them.
  StateSpaceLogitModel * CreateObservationModel(
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
  void AddData(const Vector &successes,
               const Vector &trials,
               const Matrix &predictors,
               const std::vector<bool> &response_is_observed);
 private:
  Ptr<StateSpaceLogitModel> model_;
  int predictor_dimension_;
  int clt_threshold_;

  Vector forecast_trials_;
  Matrix forecast_predictors_;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // ANALYSIS_COMMON_R_BSTS_SRC_STATE_SPACE_LOGIT_MODEL_MANAGER_H_

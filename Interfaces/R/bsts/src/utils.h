// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#ifndef ANALYSIS_COMMON_R_BSTS_SRC_UTILS_H_
#define ANALYSIS_COMMON_R_BSTS_SRC_UTILS_H_

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace BOOM {
namespace bsts {
// Factory method for creating a StateModel based on inputs supplied
// to R.  Returns a BOOM-style smart pointer to the StateModel that
// gets created.
// Args:
//   list_arg: The portion of the state.specification list (that was
//     supplied to R by the user), corresponding to the state model
//     that needs to be created
//   io_manager: A pointer to the object manaaging the R list that
//     will record (or has already recorded) the MCMC output
// Returns:
//   A BOOM smart pointer to a StateModel that can be added to a
//   StateSpaceModelBase.
Ptr<StateModel> CreateStateModel(
    SEXP list_arg, RListIoManager *io_manager);

// Looks inside the list r_object for an element with the given
// 'name'.  If the element is found, convert it to a matrix using
// ToBoomMatrix.  If not found, then return a column matrix of all 1's
// with number of rows given by default_length.
Matrix ExtractPredictors(SEXP r_object,
                         const std::string &name,
                         int default_length);

//======================================================================
// Returns a std::vector<bool> the same length as the input vector.
// Element is of the response is true iff element i of the input is
// not NA.
std::vector<bool> IsObserved(SEXP r_vector);

//======================================================================
// Record the state of a DynamicRegressionStateModel in the io_manager.
// Args:
//   model: The state space model that may or may not have a dynamic
//     regression state component to be recorded.
//   io_manager: The io_manager in charge of building the list
//     containing the dynamic regression coefficients.
void RecordDynamicRegression(StateSpaceModelBase * model,
                             RListIoManager *io_manager);

//======================================================================
// Initialize the model to be empty, except for variables that are
// known to be present with probability 1.
void DropUnforcedCoefficients(const Ptr<GlmModel> &glm,
                              const BOOM::Vector &prior_inclusion_probs);

//======================================================================
// A callback class for computing the contribution of each state model
// (including a regression component if there is one) at each time
// point.
class GeneralStateContributionCallback
    : public MatrixIoCallback {
 public:
  explicit GeneralStateContributionCallback(
      StateSpaceModelBase *model)
      : model_(model),
        has_regression_(-1) {}

  int nrow() const override {return model_->nstate() + has_regression();}
  int ncol() const override {return model_->time_dimension();}
  BOOM::Matrix get_matrix() const override {
    BOOM::Matrix ans(nrow(), ncol());
    for (int state = 0; state < model_->nstate(); ++state) {
      ans.row(state) = model_->state_contribution(state);
    }
    if (has_regression()) {
      ans.last_row() = model_->regression_contribution();
    }
    return ans;
  }

  bool has_regression() const {
    if (has_regression_ == -1) {
      Vector regression_contribution = model_->regression_contribution();
      has_regression_ = !regression_contribution.empty();
    }
    return has_regression_;
  }

 private:
  const StateSpaceModelBase *model_;
  mutable int has_regression_;
};


//======================================================================
// A callback class for saving one step ahead prediction errors from
// the Kalman filter.
class PredictionErrorCallback : public VectorIoCallback {
 public:
  explicit PredictionErrorCallback(StateSpaceModelBase *model)
      : model_(model) {}

  // Each element is a vector of one step ahead prediction errors, so
  // the dimension is the time dimension of the model.
  int dim() const override {
    return model_->time_dimension();
  }

  Vector get_vector() const override {
    return model_->one_step_prediction_errors();
  }

 private:
  StateSpaceModelBase *model_;
};

// A callback class for saving log likelihood values.
class LogLikelihoodCallback : public ScalarIoCallback {
 public:
  explicit LogLikelihoodCallback(StateSpaceModelBase *model)
      : model_(model) {}
  double get_value() const override {
    return model_->log_likelihood();
  }
 private:
  StateSpaceModelBase *model_;
};

}  // namespace bsts
}  // namespace BOOM

#endif  // ANALYSIS_COMMON_R_BSTS_SRC_UTILS_H_

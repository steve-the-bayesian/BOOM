// Copyright 2011 Google Inc. All Rights Reserved.
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

#ifndef BSTS_SRC_UTILS_H_
#define BSTS_SRC_UTILS_H_

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/list_io.hpp"
#include "LinAlg/Selector.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"

namespace BOOM {
  namespace bsts {
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
    SelectorMatrix IsObserved(const Matrix &matrix);
    
    //======================================================================
    // Record the state of a DynamicRegressionStateModel in the io_manager.
    // Args:
    //   model: The state space model that may or may not have a dynamic
    //     regression state component to be recorded.
    //   io_manager: The io_manager in charge of building the list
    //     containing the dynamic regression coefficients.
    void RecordDynamicRegression(ScalarStateSpaceModelBase * model,
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
    class ScalarStateContributionCallback
        : public MatrixIoCallback {
     public:
      // Args:
      //   model:  The model from which final state is to be obtained.
      explicit ScalarStateContributionCallback(ScalarStateSpaceModelBase *model)
          : model_(model),
            has_regression_(-1)
      {
        if (!model) {
          report_error("Null model passed to ScalarStateContributionCallback.");
        }
        if (model->state_dimension() <= 0) {
          report_error("Model has no state.");
        }
      }

      int nrow() const override {
        return model_->number_of_state_models() + has_regression();
      }
      int ncol() const override {return model_->time_dimension();}
      BOOM::Matrix get_matrix() const override {
        BOOM::Matrix ans(nrow(), ncol());
        for (int state = 0; state < model_->number_of_state_models(); ++state) {
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
      const ScalarStateSpaceModelBase *model_;
      mutable int has_regression_;
    };

    //======================================================================
    class MultivariateStateContributionCallback
        : public ArrayIoCallback {
     public:
      MultivariateStateContributionCallback(
          const MultivariateStateSpaceModelBase *model,
          int nseries)
          : model_(model),
            nseries_(nseries)
      {}
      
      std::vector<int> dim() const override {
        return {model_->number_of_state_models(),
                model_->time_dimension(),
                nseries_};
      }

      void write_to_array(ArrayView &view) const override {
        for (int s = 0; s < model_->number_of_state_models(); ++s) {
          view.slice(s, -1, -1) = model_->state_contributions(s);
        }
      }

      void read_from_array(const ArrayView &view) override {
        report_error("State contributions should not be streamed.");
      }
      
     private:
      const MultivariateStateSpaceModelBase *model_;
      const int nseries_;
    };
    
    //======================================================================
    class DynamicInterceptStateContributionCallback
        : public MatrixIoCallback {
     public:
      explicit DynamicInterceptStateContributionCallback(
          DynamicInterceptRegressionModel *model)
          : model_(model) {}

      int nrow() const override {return model_->number_of_state_models() - 1;}
      int ncol() const override {return model_->time_dimension();}

      BOOM::Matrix get_matrix() const override {
        BOOM::Matrix ans(nrow(), ncol());
        for (int state = 1; state < model_->number_of_state_models(); ++state) {
          ans.row(state - 1) = model_->state_contribution(state);
        }
        return ans;
      }
     private:
      DynamicInterceptRegressionModel *model_;
    };

    //======================================================================
    // A callback class for saving one step ahead prediction errors from
    // the Kalman filter.
    class PredictionErrorCallback : public VectorIoCallback {
     public:
      explicit PredictionErrorCallback(ScalarStateSpaceModelBase *model)
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
      ScalarStateSpaceModelBase *model_;
    };

    // A callback class for saving log likelihood values.
    class LogLikelihoodCallback : public ScalarIoCallback {
   public:
      explicit LogLikelihoodCallback(ScalarStateSpaceModelBase *model)
          : model_(model) {}
      double get_value() const override {
        return model_->log_likelihood();
      }
   private:
      ScalarStateSpaceModelBase *model_;
    };

  }  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_UTILS_H_

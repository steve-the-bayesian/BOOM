// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_DYNAMIC_INTERCEPT_REGRESSION_MODEL_HPP_
#define BOOM_DYNAMIC_INTERCEPT_REGRESSION_MODEL_HPP_
/*
  Copyright (C) 2005-2017 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/StateModels/RegressionStateModel.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Filters/ConditionalIidKalmanFilter.hpp"

namespace BOOM {

  namespace StateSpace {

    // A data type to use with DynamicInterceptRegressionModel.  Each object
    // represents the data set available at a given time point.  There is a
    // response vector and a predictor matrix, with potentially different
    // numbers of observations at each time point.
    class TimeSeriesRegressionData : public Data {
     public:
      // Args:
      //   response:  The vector of response values for this time point.
      //   predictors: The matrix of predictor variables for this time point.
      //     The number of rows must match the length of response.
      //   observed: Indicates which elements of 'response' are observed.  Its
      //     length must match the length of response.
      TimeSeriesRegressionData(const Vector &response,
                               const Matrix &predictors,
                               const Selector &observed);
      TimeSeriesRegressionData * clone() const override;

      std::ostream &display(std::ostream &out) const override;

      // Return the i'th observation at this time point.
      const Ptr<RegressionData> &regression_data(int i) const {
        return regression_data_[i];
      }
      Ptr<RegressionData> &regression_data(int i) {
        return regression_data_[i];
      }

      // The number of observations at this time point.
      int sample_size() const { return response_.size(); }

      const Vector &response() const {return response_;}
      const Matrix &predictors() const {return predictors_;}

      // All observations are 'observed' so the observation vector returns all
      // true.
      const Selector &observed() const {return observed_;}
      
     private:
      Vector response_;
      Matrix predictors_;
      std::vector<Ptr<RegressionData>> regression_data_;
      Selector observed_;
    };

  }  // namespace StateSpace

  // A DynamicInterceptRegressionModel is a time series regression model where
  // the intercept term obeys a classic state space model, but where the
  // remaining regression coefficients are static.  Note that the number of
  // observations at each time point might differ.  The model is implemented as
  // a multivariate state space model.  Through data augmentation one can extend
  // this model to most GLM's with dynamic intercepts.
  //
  // The model is Y_t^T = [y_1t, y_2t, ... y_n_tt],
  // where Y_t = Z[t] * state[t] + error[t],
  // with error[t] ~ N(0,  Diagonal(sigma^2)).
  //
  // Here Z is a matrix with one row per element of Y (which might have a
  // different number of elements for each time point).  Standard state models
  // contribute constant columns to Z (i.e. each has identical elements within
  // the same column).  However, dynamic regression models, for example will
  // contribute different predictor values for each element of Y.
  //
  // TODO: Once this works and all GLM's have been implemented, remove the
  // MultiplexedData concept from StateSpaceModelBase, which should simplify it
  // considerably.
  class DynamicInterceptRegressionModel
      : public ConditionalIidMultivariateStateSpaceModelBase,
        public IID_DataPolicy<StateSpace::TimeSeriesRegressionData>,
        public PriorPolicy {
   public:
    explicit DynamicInterceptRegressionModel(int xdim);
    DynamicInterceptRegressionModel(const DynamicInterceptRegressionModel &rhs);
    DynamicInterceptRegressionModel *clone() const override {
      return new DynamicInterceptRegressionModel(*this);
    }
    DynamicInterceptRegressionModel(DynamicInterceptRegressionModel &&rhs) =
        default;

    void add_state(const Ptr<DynamicInterceptStateModel> &state_model) {
      state_models_.push_back(state_model);
      StateSpaceModelBase::add_state(state_model);
    }
    
    RegressionModel *observation_model() override;
    const RegressionModel *observation_model() const override;
    void observe_data_given_state(int t) override;
    
    void impute_state(RNG &rng) override;
    
    int time_dimension() const override { return dat().size(); }
    int xdim() const { return regression_->regression()->xdim(); }

    bool is_missing_observation(int t) const override {
      return dat()[t]->missing() == Data::completely_missing ||
             dat()[t]->sample_size() == 0;
    }

    const Selector &observed_status(int t) const override;

    // Need to override add_data so that x's can be shared with the
    // regression model.
    void add_data(const Ptr<Data> &dp) override;

    // Adds dp to the vector of data, as the most recent observation, and adds
    // the regression data in 'dp' to the underlying regression model.
    void add_data(const Ptr<StateSpace::TimeSeriesRegressionData> &dp) override;

    void add_data(StateSpace::TimeSeriesRegressionData *dp) override;

    // Return the observation coefficients Z[t] in the observation equation:
    // y[t] = Z[t] * state[t] + error[t].  If some elements of y[t] are
    // unobserved, the dimension of Z[t] will be reduced so that Z[t] * state[t]
    // only gives the mean of the observed components.
    const SparseKalmanMatrix *observation_coefficients(int t) const override;
    
    double observation_variance(int t) const override;
    const Vector &observation(int t) const override;
    
    // Returns the conditional mean the data at time t given state and model
    // parameters.
    // Args:
    //   time:  The time index of the observation.
    Vector conditional_mean(int time) const;

    // Return the time series of contributions of each state model to the
    // intercept term.  It is an error to call this with state_model_index <= 1,
    // because the first 'state model' is the regression state model, which 
    Vector state_contribution(int state_model_index) const;
    
   private:
    // Reimplements the logic in the base class, but optimized for the scalar
    // observation variance.
    Vector simulate_observation(RNG &rng, int t) override;

    void initialize_regression_component(int xdim);

    void update_observation_model_complete_data_sufficient_statistics(
        int t, const Vector &observation_error_mean,
        const SpdMatrix &observation_error_variance) override {
      report_error(
          "EM algorithm is not yet supported for "
          "DynamicInterceptRegressionModel.");
    }

    void update_observation_model_gradient(
        VectorView gradient, int t, const Vector &state_error_mean,
        const SpdMatrix &state_error_variance) override {
      report_error(
          "MAP estimation is not yet supported for "
          "DynamicInterceptRegressionModel.");
    }

    //--------------------------------------------------------------------------
    // Data begins here.
    //--------------------------------------------------------------------------
    // The regression component of the model is the first state component.
    Ptr<RegressionDynamicInterceptStateModel> regression_;

    // This set of state parallels the state_models_ vector in the base class.
    // These are needed to provide the right version of observation_coefficients().
    std::vector<Ptr<DynamicInterceptStateModel>> state_models_;
    
    // The observation coefficients are a set of horizontal blocks
    // (i.e. vertical strips?).  Each state component contributes a block.  The
    // number of rows is the number of elements in y[t].
    mutable SparseVerticalStripMatrix observation_coefficients_;
  };

}  // namespace BOOM

#endif  // BOOM_DYNAMIC_INTERCEPT_REGRESSION_MODEL_HPP_

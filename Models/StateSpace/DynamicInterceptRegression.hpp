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
#include "Models/StateSpace/StateSpaceRegressionModel.hpp"

namespace BOOM {
  // A DynamicInterceptRegressionModel is a time series regression model where
  // the intercept term obeys a classic state space model, but where the
  // remaining regression coefficients are static.  Note that the number of
  // observations at each time point might differ.  The model is implemented as
  // a multivariate state space model.  Through data augmentation you can extend
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
      : public MultivariateStateSpaceModelBase,
        public IID_DataPolicy<StateSpace::MultiplexedRegressionData>,
        public PriorPolicy {
   public:
    explicit DynamicInterceptRegressionModel(int xdim);
    DynamicInterceptRegressionModel(const DynamicInterceptRegressionModel &rhs);
    DynamicInterceptRegressionModel *clone() const override {
      return new DynamicInterceptRegressionModel(*this);
    }
    DynamicInterceptRegressionModel(DynamicInterceptRegressionModel &&rhs) =
        default;

    RegressionModel *observation_model() override;
    const RegressionModel *observation_model() const override;
    void observe_data_given_state(int t) override;

    int time_dimension() const override { return dat().size(); }
    int xdim() const { return regression_->regression()->xdim(); }

    bool is_missing_observation(int t) const override {
      return dat()[t]->missing() == Data::completely_missing ||
             dat()[t]->observed_sample_size() == 0;
    }

    // Need to override add_data so that x's can be shared with the
    // regression model.
    void add_data(const Ptr<Data> &dp) override;

    // Adds dp to the vector of data, as the most recent observation, and adds
    // the regression data in 'dp' to the underlying regression model.
    void add_multiplexed_data(
        const Ptr<StateSpace::MultiplexedRegressionData> &dp);

    const SparseKalmanMatrix *observation_coefficients(int t) const override;
    SpdMatrix observation_variance(int t) const override;
    const Vector &observation(int t) const override { return observations_[t]; }

    // Returns the conditional mean of a specific observation at time t, given
    // state and model parameters.
    // Args:
    //   time:  The time index of the observation.
    //   observation:  The specific observation number at the time index.
    double conditional_mean(int time, int observation) const;

   protected:
    void observe_state(int t) override;

   private:
    // Reimplements the logic in the base class, but ???.
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
    Ptr<RegressionStateModel> regression_;

    // The observation coefficients are a set of horizontal blocks
    // (i.e. vertical strips?).  Each state component contributes a block.  The
    // number of rows is the number of elements in y[t].
    mutable SparseVerticalStripMatrix observation_coefficients_;

    // Element t contains the response vector for time point t.  Elements can be
    // of different sizes.
    std::vector<Vector> observations_;
  };

}  // namespace BOOM

#endif  // BOOM_DYNAMIC_INTERCEPT_REGRESSION_MODEL_HPP_

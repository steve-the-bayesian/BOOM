// Copyright 2018 Google LLC. All Rights Reserved.
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

#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using DIRM = DynamicInterceptRegressionModel;
    using StateSpace::MultiplexedRegressionData;
  }  // namespace

  DIRM::DynamicInterceptRegressionModel(int xdim) {
    initialize_regression_component(xdim);
  }

  DIRM::DynamicInterceptRegressionModel(const DIRM &rhs)
      : MultivariateStateSpaceModelBase(rhs), observations_(rhs.observations_) {
    initialize_regression_component(rhs.xdim());
    regression_->regression()->set_Beta(rhs.regression_->regression()->Beta());
    regression_->regression()->set_sigsq(
        rhs.regression_->regression()->sigsq());
  }

  RegressionModel *DIRM::observation_model() {
    return regression_->regression();
  }

  const RegressionModel *DIRM::observation_model() const {
    return regression_->regression();
  }

  void DIRM::observe_state(int t) {
    if (t == 0) {
      observe_initial_state();
      return;
    }
    const ConstVectorView now(state().col(t));
    const ConstVectorView then(state().col(t - 1));
    for (int s = 0; s < nstate(); ++s) {
      report_error(
          "Need to implement observe_dynamic_regression_state in all "
          "StateModels.");
      state_model(s)->observe_dynamic_intercept_regression_state(
          state_component(then, s), state_component(now, s), t, this);
    }
  }

  void DIRM::observe_data_given_state(int t) {
    if (!is_missing_observation(t)) {
      // Unless the data point is completely missing, add the regression
      // component of its data to the regression model.  We will do this by
      // subtracting the state mean from the y value of each observation.  The
      // state mean contains contribution from all state elements, including the
      // regression, so we need to add the regression component back in.
      Ptr<MultiplexedRegressionData> data(dat()[t]);
      Vector state_mean = (*observation_coefficients(t)) * state(t);
      RegressionModel *regression = regression_->regression();
      for (int i = 0; i < data->total_sample_size(); ++i) {
        const RegressionData &data_point(data->regression_data(i));
        double regression_prediction = regression->predict(data_point.x());
        observation_model()->suf()->add_mixture_data(
            data_point.y() - state_mean[i] + regression_prediction,
            data_point.x(), 1.0);
      }
    }
  }

  void DIRM::add_data(const Ptr<Data> &dp) { add_multiplexed_data(DAT(dp)); }

  void DIRM::add_multiplexed_data(const Ptr<MultiplexedRegressionData> &dp) {
    Vector observation_vector(dp->total_sample_size());
    Matrix predictors(dp->total_sample_size(),
                      regression_->regression()->xdim());
    for (int i = 0; i < observation_vector.size(); ++i) {
      observation_vector[i] = dp->regression_data(i).y();
      regression_->regression()->add_data(dp->regression_data_ptr(i));
      predictors.row(i) = dp->regression_data(i).x();
    }
    observations_.push_back(observation_vector);
    regression_->add_predictor_data({1, predictors});

    DataPolicy::add_data(dp);
    for (int i = 0; i < dp->total_sample_size(); ++i) {
      observation_model()->add_data(dp->regression_data_ptr(i));
    }
  }

  const SparseKalmanMatrix *DIRM::observation_coefficients(
      int t) const {
    observation_coefficients_.clear();
    for (int s = 0; s < nstate(); ++s) {
      observation_coefficients_.add_block(
          state_model(s)->dynamic_intercept_regression_observation_coefficients(
                  t, *dat()[t]));
    }
    return &observation_coefficients_;
  }

  SpdMatrix DIRM::observation_variance(int t) const {
    return SpdMatrix(dat()[t]->total_sample_size(),
                     regression_->regression()->sigsq());
  }

  double DIRM::conditional_mean(int time, int observation) const {
    report_error(
        "Need to implement DynamicInterceptRegressionModel::conditional_mean.");
    return negative_infinity();

    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
    //////////////////////////////
  }

  //===========================================================================
  // private:
  Vector DIRM::simulate_observation(RNG &rng, int t) {
    Vector ans = (*observation_coefficients(t)) * state(t);
    double residual_sd = regression_->regression()->sigma();
    for (int i = 0; i < ans.size(); ++i) {
      ans[i] += rnorm_mt(rng, 0, residual_sd);
    }
    return ans;
  }

  void DIRM::initialize_regression_component(int xdim) {
    regression_.reset(new RegressionStateModel(new RegressionModel(xdim)));
    add_state(regression_);
  }

}  // namespace BOOM

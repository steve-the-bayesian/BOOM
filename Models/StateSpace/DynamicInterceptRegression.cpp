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
    using StateSpace::TimeSeriesRegressionData;
  }  // namespace

  namespace StateSpace {
    TimeSeriesRegressionData::TimeSeriesRegressionData(
        const Vector &response,
        const Matrix &predictors,
        const Selector &observed)
        : response_(response),
          predictors_(predictors),
          observed_(observed)
    {
      if (response_.size() != predictors_.nrow()) {
        report_error("Different numbers of observations in 'response' "
                     "and 'predictors'.");
      }
      if (response_.size() != observed.nvars_possible()) {
        report_error("Observation indicator and response vector must "
                     "be the same size.");
      }
      for (int i = 0; i < response_.size(); ++i) {
        regression_data_.push_back(new RegressionData(
            response_[i], predictors_.row(i)));
      }
    }

    TimeSeriesRegressionData * TimeSeriesRegressionData::clone() const {
      return new TimeSeriesRegressionData(*this);
    }

    std::ostream &TimeSeriesRegressionData::display(std::ostream &out) const {
      out << cbind(response_, predictors_);
      return out;
    }

  }  // namespace StateSpace

  //===========================================================================

  DIRM::DynamicInterceptRegressionModel(int xdim) {
    initialize_regression_component(xdim);
  }

  DIRM::DynamicInterceptRegressionModel(const DIRM &rhs)
      : ConditionalIidMultivariateStateSpaceModelBase(rhs) {
    initialize_regression_component(rhs.xdim());
    regression_->regression()->set_Beta(rhs.regression_->regression()->Beta());
    regression_->regression()->set_sigsq(
        rhs.regression_->regression()->sigsq());
    for (int s = 0; s < rhs.number_of_state_models(); ++s) {
      add_state(rhs.state_model(s)->clone());
    }
  }

  RegressionModel *DIRM::observation_model() {
    return regression_->regression();
  }

  const RegressionModel *DIRM::observation_model() const {
    return regression_->regression();
  }

  void DIRM::observe_data_given_state(int t) {
    const Selector &observed(observed_status(t));
    if (observed.nvars() > 0) {
      // Unless the data point is completely missing, add the regression
      // component of its data to the regression model.  We will do this by
      // subtracting the state mean from the y value of each observation.  The
      // state mean contains contribution from all state elements, including the
      // regression, so we need to add the regression component back in.
      Ptr<TimeSeriesRegressionData> data(dat()[t]);
      Vector state_contribution = (*observation_coefficients(
          t, observed_status(t))) * shared_state(t);

      RegressionModel *regression = regression_->regression();
      for (int i = 0; i < data->sample_size(); ++i) {
        const Ptr<RegressionData> &data_point(data->regression_data(i));
        double adjusted_observation =
            data_point->y() - state_contribution[i]
            + regression->predict(data_point->x());
        observation_model()->suf()->add_mixture_data(
            adjusted_observation, data_point->x(), 1.0);
      }
    }
  }

  void DIRM::observe_state(int t) {
    if (t == 0) {
      for (int s = 0; s < state_models_.size(); ++s) {
        state_model(s)->observe_initial_state(
            state_models_.state_component(shared_state().col(0), s));
      }
    } else {
      const ConstVectorView now(shared_state().col(t));
      const ConstVectorView then(shared_state().col(t - 1));
      for (int s = 0; s < state_models_.size(); ++s) {
        state_models_[s]->observe_state(
            state_models_.state_component(then, s),
            state_models_.state_component(now, s),
            t);
      }
    }
  }

  void DIRM::impute_state(RNG &rng) {
    MultivariateStateSpaceModelBase::impute_state(rng);
    observation_model()->suf()->fix_xtx();
  }

  void DIRM::add_data(const Ptr<Data> &dp) { add_data(DAT(dp)); }
  void DIRM::add_data(TimeSeriesRegressionData *dp) {
    add_data(Ptr<TimeSeriesRegressionData>(dp));
  }
  void DIRM::add_data(const Ptr<TimeSeriesRegressionData> &dp) {
    for (int i = 0; i < dp->sample_size(); ++i) {
      regression_->regression()->add_data(dp->regression_data(i));
    }
    DataPolicy::add_data(dp);
  }

  Ptr<SparseKalmanMatrix> DIRM::observation_coefficients(
      int t, const Selector &) const {
    NEW(SparseVerticalStripMatrix, ans)();
    const StateSpace::TimeSeriesRegressionData &data_point(*dat()[t]);
    for (int s = 0; s < number_of_state_models(); ++s) {
      ans->add_block(state_models_[s]->observation_coefficients(t, data_point));
    }
    return ans;
  }

  double DIRM::observation_variance(int t) const {
    return regression_->regression()->sigsq();
  }

  ConstVectorView DIRM::observation(int t) const {
    return dat()[t]->response();
  }

  ConstVectorView DIRM::adjusted_observation(int time) const {
    return observation(time);
  }

  const Selector &DIRM::observed_status(int t) const {
    return dat()[t]->observed();
  }

  Vector DIRM::conditional_mean(int time) const {
    return (*observation_coefficients(
        time, observed_status(time)) * shared_state().col(time));
  }

  Vector DIRM::state_contribution(int state_model_index) const {
    if (state_model_index == 0) {
      report_error("You can't use a Vector summarize the state contribution "
                   "from the regression component because there can be more "
                   "than one observation per time period.");
    } else if (state_model_index < 0) {
      report_error("state_model_index must be at least 1.");
    } else if (state_model_index >= number_of_state_models()) {
      report_error("state_model_index too large.");
    } else if (!state_models_[state_model_index]->is_pure_function_of_time()) {
      std::ostringstream err;
      err << "The model in position " << state_model_index
          << " is not a pure function of time.";
      report_error(err.str());
    }

    Vector ans(time_dimension());
    const Matrix &state(this->shared_state());
    TimeSeriesRegressionData dummy_data(
        Vector(1, 0.0), Matrix(1, 1, 0.0), Selector(1, true));
    for (int t = 0; t < time_dimension(); ++t) {
      ConstVectorView local_state(
          state_models_.state_component(state.col(t), state_model_index));
      Vector tmp = *state_model(state_model_index)->observation_coefficients(
          t, dummy_data) * local_state;
      ans[t] = tmp[0];
    }
    return ans;
  }

  Vector DIRM::simulate_forecast(RNG &rng,
                                 const Matrix &forecast_predictors,
                                 const Vector &final_state,
                                 const std::vector<int> &timestamps) {
    if (nrow(forecast_predictors) != timestamps.size()) {
      report_error("different numbers of timestamps and forecast_predictors.");
    }
    if (final_state.size() != state_dimension()) {
      std::ostringstream err;
      err << "final state argument was of dimension " << final_state.size()
          << " but model state dimension is " << state_dimension()
          << "." << std::endl;
      report_error(err.str());
    }
    Vector ans(timestamps.size());
    int t0 = time_dimension();
    int time = -1;
    Vector state = final_state;
    int index = 0;
    int xdim = ncol(forecast_predictors);

    // Move the state to the next time stamp.
    // Simulate observations for all the data with that timestamp.
    while(index < timestamps.size() && time < timestamps[index]) {
      advance_to_timestamp(rng, time, state, timestamps[index], index);
      while (index < timestamps.size() && time == timestamps[index]) {
        TimeSeriesRegressionData data_point(
            Vector(1, 0.0),
            Matrix(1, xdim, forecast_predictors.row(index)),
            Selector(1, true));
        Vector yhat = *observation_coefficients(
            t0 + time, data_point.observed()) * state;
        double sigma = sqrt(observation_variance(t0 + time));
        ans[index] = yhat[0] + rnorm_mt(rng, 0, sigma);
        ++index;
      }
    }
    return ans;
  }

  //===========================================================================
  // private:
  Vector DIRM::simulate_fake_observation(RNG &rng, int t) {
    int number_of_observations = dat()[t]->sample_size();
    Selector fully_observed(number_of_observations, true);
    const Selector &observed(
        t >= time_dimension() ? fully_observed : observed_status(t));
    Vector ans = (*observation_coefficients(t, observed)) * shared_state(t);
    double residual_sd = sqrt(observation_variance(t));
    for (int i = 0; i < ans.size(); ++i) {
      ans[i] += rnorm_mt(rng, 0, residual_sd);
    }
    return ans;
  }

  void DIRM::initialize_regression_component(int xdim) {
    regression_.reset(new RegressionDynamicInterceptStateModel(
        new RegressionModel(xdim)));
    add_state(regression_);
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  namespace {
    typedef StateSpaceModel SSM;
    typedef StateSpace::MultiplexedDoubleData MDD;
  }  // namespace

  MDD::MultiplexedDoubleData() {}

  MDD::MultiplexedDoubleData(double y) { add_data(new DoubleData(y)); }

  MDD *MDD::clone() const { return new MDD(*this); }

  std::ostream &MDD::display(std::ostream &out) const {
    for (int i = 0; i < data_.size(); ++i) {
      data_[i]->display(out) << std::endl;
    }
    return out;
  }

  void MDD::add_data(const Ptr<DoubleData> &data_point) {
    MultiplexedData::add_data(data_point);
    data_.push_back(data_point);
  }

  double MDD::adjusted_observation() const {
    if (data_.empty() || missing() == Data::completely_missing ||
        observed_sample_size() == 0) {
      return negative_infinity();
    }
    double ans = 0;
    for (int i = 0; i < data_.size(); ++i) {
      if (data_[i]->missing() == Data::observed) {
        ans += data_[i]->value();
      }
    }
    return ans / observed_sample_size();
  }

  const DoubleData &MDD::double_data(int i) const { return *(data_[i]); }

  Ptr<DoubleData> MDD::double_data_ptr(int i) { return data_[i]; }

  void MDD::set_value(double value, int i) { data_[i]->set(value); }

  bool MDD::all_missing() const {
    if (data_.empty()) return true;
    for (int i = 0; i < data_.size(); ++i) {
      if (data_[i]->missing() != Data::completely_missing) {
        return false;
      }
    }
    return true;
  }

  //======================================================================
  void SSM::setup() {
    observation_model_->only_keep_sufstats();
  }

  SSM::StateSpaceModel() : observation_model_(new ZeroMeanGaussianModel) {
    setup();
  }

  SSM::StateSpaceModel(const Vector &y, const std::vector<bool> &y_is_observed)
      : observation_model_(
            new ZeroMeanGaussianModel(sqrt(var(y, y_is_observed)) / 10)) {
    setup();
    for (int i = 0; i < y.size(); ++i) {
      NEW(MDD, dp)(y[i]);
      if (!y_is_observed.empty() && !y_is_observed[i]) {
        dp->set_missing_status(Data::completely_missing);
        dp->double_data_ptr(0)->set_missing_status(Data::completely_missing);
      }
      add_data(dp);
    }
  }

  SSM::StateSpaceModel(const SSM &rhs)
      : Model(rhs),
        ScalarStateSpaceModelBase(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        observation_model_(rhs.observation_model_->clone()) {
    setup();
  }

  SSM *SSM::clone() const { return new SSM(*this); }

  int SSM::time_dimension() const {
    return dat().size();
  }

  double SSM::observation_variance(int t) const {
    double sigsq = observation_model_->sigsq();
    if (t >= dat().size()) {
      return sigsq;
    }
    const Ptr<MDD> &data_point(dat()[t]);
    if (is_missing_observation(t) || data_point->observed_sample_size() <= 1) {
      return sigsq;
    } else {
      return sigsq / data_point->observed_sample_size();
    }
  }

  double SSM::adjusted_observation(int t) const {
    return dat()[t]->adjusted_observation();
  }

  bool SSM::is_missing_observation(int t) const {
    return dat()[t]->missing() == Data::completely_missing ||
           dat()[t]->observed_sample_size() == 0;
  }

  ZeroMeanGaussianModel *SSM::observation_model() {
    return observation_model_.get();
  }

  const ZeroMeanGaussianModel *SSM::observation_model() const {
    return observation_model_.get();
  }

  void SSM::observe_data_given_state(int t) {
    // Assuming ignorable missing data.
    if (!is_missing_observation(t)) {
      const Ptr<MDD> &data_point(dat()[t]);
      double mu = observation_matrix(t).dot(state(t));
      for (int j = 0; j < data_point->total_sample_size(); ++j) {
        if (data_point->double_data(j).missing() == Data::observed) {
          double residual = data_point->double_data(j).value() - mu;
          observation_model_->suf()->update_raw(residual);
        }
      }
    }
  }

  // TODO: should observation_matrix and observation_variance be called with t +
  // t0 + 1?
  Matrix SSM::forecast(int n) {
    // TODO: This method only works with truly Gaussian
    // state models.  We should put in a check to make sure that none
    // of the state models are T, normal mixture, etc.
    kalman_filter();
    Kalman::ScalarMarginalDistribution marg = get_filter().back();
    Matrix ans(n, 2);
    int t0 = time_dimension();
    for (int t = 0; t < n; ++t) {
      ans(t, 0) = observation_matrix(t + t0).dot(marg.state_mean());
      marg.update(0, true, t + t0);
      ans(t, 1) = sqrt(marg.prediction_variance());
    }
    return ans;
  }

  Vector SSM::simulate_forecast(RNG &rng, int n, const Vector &final_state) {
    ScalarStateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Vector ans(n);
    int t0 = time_dimension();
    Vector state = final_state;
    for (int t = 0; t < n; ++t) {
      state = simulate_next_state(rng, state, t + t0);
      ans[t] = rnorm_mt(rng, observation_matrix(t + t0).dot(state),
                        sqrt(observation_variance(t + t0)));
    }
    return ans;
  }

  Matrix SSM::simulate_forecast_components(
      RNG &rng, int forecast_horizon, const Vector &final_state) {
    ScalarStateSpaceModelBase::set_state_model_behavior(StateModel::MARGINAL);
    Matrix ans(number_of_state_models() + 1, forecast_horizon, 0.0);
    int t0 = time_dimension();
    Vector state = final_state;
    for (int t = 0; t < forecast_horizon; ++t) {
      state = simulate_next_state(rng, state, t + t0);
      for (int s = 0; s < number_of_state_models(); ++s) {
        ans(s, t) = state_model(s)->observation_matrix(t + t0).dot(
            state_component(state, s));
      }
      ans.col(t).back() = rnorm_mt(rng,
                                   ans.col(t).sum(),
                                   observation_variance(t + t0));
    }
    return ans;
  }

  Vector SSM::one_step_holdout_prediction_errors(
      const Vector &newY, const Vector &final_state, bool standardize) const {
    Vector ans(length(newY));
    int t0 = time_dimension();
    Kalman::ScalarMarginalDistribution marg(this, nullptr, 0);
    marg.set_state_mean(*state_transition_matrix(t0 - 1) * final_state);
    marg.set_state_variance(SpdMatrix(state_variance_matrix(t0 - 1)->dense()));
    for (int t = 0; t < ans.size(); ++t) {
      bool missing = false;
      marg.update(newY[t], missing, t + t0);
      ans[t] = marg.prediction_error();
      if (standardize) {
        ans[t] /= sqrt(marg.prediction_variance());
      }
    }
    return ans;
  }

  Matrix SSM::simulate_holdout_prediction_errors(
      int niter, int cutpoint_number, bool standardize) {
    Matrix ans(niter, time_dimension());
    SubMatrix training_prediction_errors(
        ans, 0, niter - 1, 0, cutpoint_number - 1);
    SubMatrix holdout_prediction_errors(
        ans, 0, niter - 1, cutpoint_number, ncol(ans) - 1);
    std::vector<Ptr<Data>> training_data(dat().begin(), dat().begin() + cutpoint_number);
    std::vector<Ptr<StateSpace::MultiplexedDoubleData>> holdout_data(
        dat().begin() + cutpoint_number, dat().end());
    clear_data();
    for (const auto &data_point : training_data) {
      add_data(data_point);
    }
    Vector holdout_data_vector;
    for (const auto &data_point : holdout_data) {
      if (data_point->total_sample_size() != 1) {
        report_error("Can't compute holdout prediction errors for "
                     "multiplex data.");
      }
      holdout_data_vector.push_back(data_point->double_data(0).value());
    }

    sample_posterior();
    for (int i = 0; i < niter; ++i) {
      sample_posterior();
      training_prediction_errors.row(i) =
          one_step_prediction_errors(standardize);
      holdout_prediction_errors.row(i) =
          one_step_holdout_prediction_errors(
              holdout_data_vector,
              state().last_col(),
              standardize);
    }

    // Replace the holdout data.
    for (const auto &data_point : holdout_data) {
      add_data(data_point);
    }
    return ans;
  }

  void SSM::update_observation_model_complete_data_sufficient_statistics(
      int t, double observation_error_mean, double observation_error_variance) {
    observation_model()->suf()->update_expected_value(
        1.0, observation_error_mean,
        observation_error_variance + square(observation_error_mean));
  }

  void SSM::update_observation_model_gradient(
      VectorView gradient, int t, double observation_error_mean,
      double observation_error_variance) {
    double sigsq = observation_model()->sigsq();
    gradient[0] +=
        (-.5 / sigsq) +
        .5 * (observation_error_variance + square(observation_error_mean)) /
            square(sigsq);
  }

}  // namespace BOOM

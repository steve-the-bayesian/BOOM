/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include <functional>

#include "LinAlg/SubMatrix.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "numopt.hpp"
#include "numopt/Powell.hpp"
#include "stats/moments.hpp"

namespace BOOM {

  namespace {
    using MvBase = MultivariateStateSpaceModelBase;
  }  // namespace

  MvBase &MvBase::operator=(const MvBase &rhs) {
    if (&rhs != this) {
      report_error("Still need top implement MultivariateStateSpaceModelBase::operator=");
      shared_state_ = rhs.shared_state_;
      state_is_fixed_ = rhs.state_is_fixed_;
    }
    return *this;
  }

  //----------------------------------------------------------------------
  void MvBase::set_state_model_behavior(StateModelBase::Behavior behavior) {
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->set_behavior(behavior);
    }
  }

  void MvBase::impute_state(RNG &rng) {
    if (number_of_state_models() == 0) {
      report_error("No state has been defined.");
    }
    set_state_model_behavior(StateModel::MIXTURE);
    if (state_is_fixed_) {
      observe_fixed_state();
    } else {
      resize_state();
      clear_client_data();
      simulate_forward(rng);
      propagate_disturbances(rng);
    }
  }

  //----------------------------------------------------------------------
  // Simulate alpha_+ and y_* = y - y_+.  While simulating y_*,
  // feed it into the light (no storage for P) Kalman filter. The
  // simulated state is stored in state_.
  //
  // y_+ and alpha_+ will be simulated in parallel with
  // Kalman filtering and disturbance smoothing of y, and the results
  // will be subtracted to compute y_*.
  void MvBase::simulate_forward(RNG &rng) {
    // Filter the observed data.
    get_filter().update();

    // Simulate and filter the fake data.
    MultivariateKalmanFilterBase &simulation_filter(get_simulation_filter());
    SpdMatrix simulated_data_state_variance = initial_state_variance();
    for (int t = 0; t < time_dimension(); ++t) {
      // simulate_state at time t
      if (t == 0) {
        simulate_initial_state(rng, shared_state_.col(0));
      } else {
        shared_state_.col(t) = simulate_next_state(
            rng, ConstVectorView(shared_state_.col(t - 1)), t);
      }
      Vector simulated_observation = observed_status(t).expand(
          simulate_fake_observation(rng, t));
      simulation_filter.update_single_observation(
          simulated_observation, observed_status(t), t);
    }
  }

  void MvBase::simulate_initial_state(RNG &rng, VectorView state0) const {
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->simulate_initial_state(
          rng, state_component(state0, s));
    }
  }

  Vector MvBase::simulate_next_state(RNG &rng, const ConstVectorView &last,
                                     int t) const {
    return (*state_transition_matrix(t - 1)) * last
        + simulate_state_error(rng, t - 1);
  }

  Vector MvBase::simulate_state_error(RNG &rng, int t) const {
    Vector ans(state_dimension());
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->simulate_state_error(rng, state_component(ans, s), t);
    }
    return ans;
  }

  void MvBase::advance_to_timestamp(RNG &rng, int &time, Vector &state,
                                    int timestamp, int observation_index) const {
    while (time < timestamp) {
      state = simulate_next_state(rng, state, time_dimension() + time++);
    }
    if (time != timestamp) {
      std::ostringstream err;
      err << "Timestamps out of order for observation " << observation_index
          << " with time = " << time << " and timestamps[" << observation_index
          << "] = " << timestamp << ".";
      report_error(err.str());
    }
  }

  Vector MvBase::initial_state_mean() const {
    Vector ans;
    for (int s = 0; s < number_of_state_models(); ++s) {
      ans.concat(state_model(s)->initial_state_mean());
    }
    return ans;
  }

  SpdMatrix MvBase::initial_state_variance() const {
    SpdMatrix ans(state_dimension());
    int lo = 0;
    for (int s = 0; s < number_of_state_models(); ++s) {
      int hi = lo + state_model(s)->state_dimension() - 1;
      SubMatrix block(ans, lo, hi, lo, hi);
      block = state_model(s)->initial_state_variance();
      lo = hi + 1;
    }
    return ans;
  }

  // AFTER a call to fast_disturbance_smoother() puts r[t] in
  // filter_[t].scaled_state_error(), this function propagates the r's forward
  // to get E(alpha | y), and add it to the simulated state.
  void MvBase::propagate_disturbances(RNG &rng) {
    if (time_dimension() <= 0) return;
    SpdMatrix P0 = initial_state_variance();
    MultivariateKalmanFilterBase &simulation_filter(
        get_simulation_filter());
    simulation_filter.fast_disturbance_smooth();
    MultivariateKalmanFilterBase &filter(get_filter());
    filter.fast_disturbance_smooth();

    Vector state_mean_sim = initial_state_mean()
        + P0 * simulation_filter.initial_scaled_state_error();
    Vector state_mean_obs = initial_state_mean()
        + P0 * filter.initial_scaled_state_error();

    shared_state_.col(0) += state_mean_obs - state_mean_sim;
    impute_missing_observations(0, rng);
    observe_state(0);
    observe_data_given_state(0);

    for (int t = 1; t < time_dimension(); ++t) {
      state_mean_sim = (*state_transition_matrix(t - 1)) * state_mean_sim +
          (*state_variance_matrix(t - 1)) *
          simulation_filter[t - 1].scaled_state_error();
      state_mean_obs =
          (*state_transition_matrix(t - 1)) * state_mean_obs +
          (*state_variance_matrix(t - 1)) * filter[t - 1].scaled_state_error();

      shared_state_.col(t).axpy(state_mean_obs - state_mean_sim);
      impute_missing_observations(t, rng);
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  //----------------------------------------------------------------------
  void MvBase::clear_client_data() {
    observation_model()->clear_data();
    state_model_vector().clear_data();
  }
  //----------------------------------------------------------------------
  void MvBase::observe_fixed_state() {
    clear_client_data();
    for (int t = 0; t < time_dimension(); ++t) {
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  void MvBase::permanently_set_state(const Matrix &state) {
    if ((ncol(state) != time_dimension()) ||
        (nrow(state) != state_dimension())) {
      ostringstream err;
      err << "Wrong dimension of 'state' in permanently_set_state()."
          << "Argument was " << nrow(state) << " by " << ncol(state)
          << ".  Expected " << state_dimension() << " by " << time_dimension()
          << "." << endl;
      report_error(err.str());
    }
    state_is_fixed_ = true;
    shared_state_ = state;
  }

  // Ensure that state_ is large enough to hold the results of
  // impute_state().
  void MvBase::resize_state() {
    if (nrow(shared_state_) != state_dimension() ||
        ncol(shared_state_) != time_dimension()) {
      shared_state_.resize(state_dimension(), time_dimension());
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->observe_time_dimension(time_dimension());
    }
  }

  //===========================================================================

  namespace {
    using CiidBase = ConditionalIidMultivariateStateSpaceModelBase;
  }

  CiidBase::ConditionalIidMultivariateStateSpaceModelBase()
      : filter_(this),
        simulation_filter_(this)
  {}

  // A precondition is that the state at time t was simulated by the forward
  // portion of the Durbin-Koopman data augmentation algorithm.
  Vector CiidBase::simulate_fake_observation(RNG &rng, int t) {
     Vector ans = (*observation_coefficients(
         t, observed_status(t))) * shared_state().col(t);
     double sigma = sqrt(observation_variance(t));
     for (int i = 0; i < ans.size(); ++i) {
       ans[i] += rnorm_mt(rng, 0, sigma);
     }
     return ans;
   }

  ConditionalIidKalmanFilter & CiidBase::get_filter() {
    return filter_;
  }

  const ConditionalIidKalmanFilter & CiidBase::get_filter() const {
    return filter_;
  }

  ConditionalIidKalmanFilter & CiidBase::get_simulation_filter() {
    return simulation_filter_;
  }

  const ConditionalIidKalmanFilter & CiidBase::get_simulation_filter() const {
    return simulation_filter_;
  }

  //===========================================================================

  namespace {
    using CindBase = ConditionallyIndependentMultivariateStateSpaceModelBase;
  }  // namespace

  Vector CindBase::simulate_fake_observation(RNG &rng, int t) {
    Vector ans = (*observation_coefficients(t, observed_status(t)))
        * shared_state().col(t);
    for (int i = 0; i < ans.size(); ++i) {
      double sigma = sqrt(single_observation_variance(t, i));
      ans[i] += rnorm_mt(rng, 0, sigma);
    }
    return ans;
  }

}  // namespace BOOM

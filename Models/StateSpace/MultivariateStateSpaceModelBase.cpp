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
      StateSpaceModelBase::operator=(rhs);
      r0_sim_ = rhs.r0_sim_;
      r0_obs_ = rhs.r0_obs_;
    }
    return *this;
  }
  
  //----------------------------------------------------------------------
  // Simulate alpha_+ and y_* = y - y_+.  While simulating y_*,
  // feed it into the light (no storage for P) Kalman filter.  The
  // simulated state is stored in state_.
  //
  // y_+ and alpha_+ will be simulated in parallel with
  // Kalman filtering and disturbance smoothing of y, and the results
  // will be subtracted to compute y_*.
  void MvBase::simulate_forward(RNG &rng) {
    MultivariateKalmanFilterBase &filter(get_filter());
    filter.update();
    MultivariateKalmanFilterBase &simulation_filter(get_simulation_filter());
    Vector simulated_data_state_mean = initial_state_mean();
    SpdMatrix simulated_data_state_variance = initial_state_variance();
    for (int t = 0; t < time_dimension(); ++t) {
      // simulate_state at time t
      if (t == 0) {
        simulate_initial_state(rng, mutable_state().col(0));
      } else {
        simulate_next_state(rng, mutable_state().col(t - 1),
                            mutable_state().col(t), t);
      }
      Vector simulated_observation = simulate_fake_observation(rng, t);
      simulation_filter.update_single_observation(
          simulated_observation, observed_status(t), t);
    }
  }

  // // After a call to fast_disturbance_smoother() puts r[t] in
  // // filter_[t].scaled_state_error(), this function propagates the r's forward
  // // to get E(alpha | y), and add it to the simulated state.
  // void MvBase::propagate_disturbances() {
  //   if (time_dimension() <= 0) return;
  //   SpdMatrix P0 = initial_state_variance();
  //   Vector state_mean_sim = initial_state_mean() + P0 * r0_sim();
  //   Vector state_mean_obs = initial_state_mean() + P0 * r0_obs();

  //   mutable_state().col(0) += state_mean_obs - state_mean_sim;
  //   observe_state(0);
  //   observe_data_given_state(0);

  //   const MultivariateKalmanFilterBase &simulation_filter(
  //       get_simulation_filter());
  //   const MultivariateKalmanFilterBase &filter(get_filter());
    
  //   for (int t = 1; t < time_dimension(); ++t) {
  //     state_mean_sim = (*state_transition_matrix(t - 1)) * state_mean_sim +
  //         (*state_variance_matrix(t - 1)) *
  //         simulation_filter[t - 1].scaled_state_error();
  //     state_mean_obs =
  //         (*state_transition_matrix(t - 1)) * state_mean_obs +
  //         (*state_variance_matrix(t - 1)) * filter[t - 1].scaled_state_error();

  //     mutable_state().col(t).axpy(state_mean_obs - state_mean_sim);
  //     observe_state(t);
  //     observe_data_given_state(t);
  //   }
  // }

  void MvBase::update_observation_model(Vector &r, SpdMatrix &N, int t,
                                        bool save_state_distributions,
                                        bool update_sufficient_statistics,
                                        Vector *gradient) {
    report_error("MAP estimation is not yet supported for multivariate models");
  }
  
  //===========================================================================

  namespace {
    using CiidBase = ConditionalIidMultivariateStateSpaceModelBase;
  }

  CiidBase::ConditionalIidMultivariateStateSpaceModelBase(int nseries)
      : MvBase(nseries),
        filter_(this),
        simulation_filter_(this)
  {}
  
  // A precondition is that the state at time t was simulated by the forward
  // portion of the Durbin-Koopman data augmentation algorithm.
  Vector CiidBase::simulate_fake_observation(RNG &rng, int t) {
     Vector ans = (*observation_coefficients(
         t, observed_status(t))) * state().col(t);
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
        * state().col(t);
    for (int i = 0; i < ans.size(); ++i) {
      double sigma = sqrt(single_observation_variance(t, i));
      ans[i] += rnorm_mt(rng, 0, sigma);
    }
    return ans;
  }

}  // namespace BOOM

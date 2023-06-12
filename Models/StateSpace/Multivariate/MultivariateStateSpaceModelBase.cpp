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

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include <functional>

#include "LinAlg/SubMatrix.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "numopt.hpp"
#include "numopt/Powell.hpp"
#include "stats/moments.hpp"

namespace BOOM {


  namespace {
    using MvBase = MultivariateStateSpaceModelBase;

    // A functor that computes the log likelihood of a MvBase under a candidate
    // set of parameters.
    class MultivariateStateSpaceTargetFun {
     public:
      explicit MultivariateStateSpaceTargetFun(MvBase *model)
          : model_(model) {}

      double operator()(const Vector &parameters) {
        Vector old_parameters(model_->vectorize_params());
        model_->unvectorize_params(parameters);
        double ans = model_->log_likelihood();
        model_->unvectorize_params(old_parameters);
        return ans;
      }

     private:
      MvBase *model_;
    };
  }  // namespace

  MvBase &MvBase::operator=(const MvBase &rhs) {
    if (&rhs != this) {
      report_error("Still need top implement MultivariateStateSpaceModelBase::operator=");
      shared_state_ = rhs.shared_state_;
      state_is_fixed_ = rhs.state_is_fixed_;
      show_warnings_ = rhs.show_warnings_;
    }
    return *this;
  }


  // Copy the posterior samplers from rhs.
  void MvBase::copy_samplers(const MvBase &rhs) {
    clear_methods();
    observation_model()->clear_methods();
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->clear_methods();
    }

    int num_methods = rhs.observation_model()->number_of_sampling_methods();
    for (int m = 0; m < num_methods; ++m) {
      observation_model()->set_method(
          rhs.observation_model()->sampler(m)->clone_to_new_host(
              observation_model()));
    }

    for (int s = 0; s < number_of_state_models(); ++s) {
      num_methods = rhs.state_model(s)->number_of_sampling_methods();
      for (int m = 0; m < num_methods; ++m) {
        state_model(s)->set_method(
            rhs.state_model(s)->sampler(m)->clone_to_new_host(
                state_model(s)));
      }
    }

    num_methods =rhs.number_of_sampling_methods();
    for (int m = 0; m < num_methods; ++m) {
      set_method(rhs.sampler(m)->clone_to_new_host(this));
    }
  }

  //----------------------------------------------------------------------
  namespace {
    // Return a single std::vector of parameter vectors formed by concatenating
    // a collection of such vectors.
    std::vector<Ptr<Params>> concatenate_parameter_vectors(
        const std::vector<std::vector<Ptr<Params>>> &vectors) {
      std::vector<Ptr<Params>> ans;
      for (const auto &v : vectors) {
        for (const auto &el : v) {
          ans.push_back(el);
        }
      }
      return ans;
    }
  }  // namespace


  std::vector<Ptr<Params>> MvBase::parameter_vector() {
    std::vector<std::vector<Ptr<Params>>> vectors;
    vectors.push_back(observation_model()->parameter_vector());
    for (int s = 0; s < number_of_state_models(); ++s) {
      vectors.push_back(state_model(s)->parameter_vector());
    }
    return concatenate_parameter_vectors(vectors);
  }

  const std::vector<Ptr<Params>> MvBase::parameter_vector() const {
    std::vector<std::vector<Ptr<Params>>> vectors;
    vectors.push_back(observation_model()->parameter_vector());
    for (int s = 0; s < number_of_state_models(); ++s) {
      vectors.push_back(state_model(s)->parameter_vector());
    }
    return concatenate_parameter_vectors(vectors);
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
  // simulated state is stored in shared_state_.
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
      Vector simulated_observation = simulate_fake_observation(rng, t);
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

    MultivariateKalmanFilterBase &filter(get_filter());
    filter.fast_disturbance_smooth();
    MultivariateKalmanFilterBase &simulation_filter(get_simulation_filter());
    simulation_filter.fast_disturbance_smooth();

    SpdMatrix P0 = initial_state_variance();
    Vector state_mean_sim = initial_state_mean()
        + P0 * simulation_filter.initial_scaled_state_error();
    Vector state_mean_obs = initial_state_mean()
        + P0 * filter.initial_scaled_state_error();

    shared_state_.col(0) += state_mean_obs - state_mean_sim;
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
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  //----------------------------------------------------------------------
  void MvBase::clear_client_data() {
    observation_model()->clear_data();
    state_models().clear_data();
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

      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->observe_time_dimension(time_dimension());
      }
    }
  }

  VectorView MvBase::state_parameter_component(Vector &model_parameters,
                                               int s) const {
    if (observation_model_parameter_size_ < 0) {
      observation_model_parameter_size_ =
          observation_model()->vectorize_params().size();
    }
    int start = observation_model_parameter_size_
        + state_models().state_parameter_position(s);
    int size = state_models().state_parameter_size(s);
    return VectorView(model_parameters, start, size);
  }

  ConstVectorView MvBase::state_parameter_component(
      const Vector &model_parameters, int s) const {
    if (observation_model_parameter_size_ < 0) {
      observation_model_parameter_size_ =
          observation_model()->vectorize_params().size();
    }
    int start = observation_model_parameter_size_
        + state_models().state_parameter_position(s);
    int size = state_models().state_parameter_size(s);
    return ConstVectorView(model_parameters, start, size);
  }

  double MvBase::mle(double epsilon, int max_tries) {
    // If the model can be estimated using an EM algorithm, then do a
    // few steps of EM, and then switch to BFGS.
    MultivariateStateSpaceTargetFun target(this);
    Negate min_target(target);
    PowellMinimizer minimizer(min_target);
    minimizer.set_evaluation_limit(max_tries);
    Vector parameters = vectorize_params(true);
    minimizer.set_precision(epsilon);
    minimizer.minimize(parameters);
    unvectorize_params(minimizer.minimizing_value());
    return log_likelihood();
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
    const Selector &obs(observed_status(t));
    if (obs.nvars() == 0) {
      return Vector(0);
    } else {
      Vector ans = (*observation_coefficients(t, obs) * shared_state().col(t));
      double sigma = sqrt(observation_variance(t));
      for (int i = 0; i < ans.size(); ++i) {
        ans[i] += rnorm_mt(rng, 0, sigma);
      }
      return ans;
    }
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

  void CiidBase::update_observation_model(
      Vector &r,
      SpdMatrix &N,
      int t,
      bool save_state_distributions,
      bool update_sufficient_statistics,
      Vector *gradient) {
    report_error("CiidBase::update_observation_model is not implemented.");
  }

  //===========================================================================

  namespace {
    using CindBase = ConditionallyIndependentMultivariateStateSpaceModelBase;
  }  // namespace

  Vector CindBase::simulate_fake_observation(RNG &rng, int t) {
    const Selector &obs(observed_status(t));
    if (obs.nvars() == 0) {
      return Vector(0);
    } else {
      Vector ans = (*observation_coefficients(t, obs)) * shared_state().col(t);
      for (int i = 0; i < ans.size(); ++i) {
        double sigma = sqrt(single_observation_variance(t, i));
        ans[i] += rnorm_mt(rng, 0, sigma);
      }
      return ans;
    }
  }

  void CindBase::update_observation_model(
      Vector &r,
      SpdMatrix &N,
      int t,
      bool save_state_distributions,
      bool update_sufficient_statistics,
      Vector *gradient) {

    Kalman::ConditionallyIndependentMarginalDistribution &marg(get_filter()[t]);
    // Some syntactic sugar to make later formulas easier to read.  These are
    // bad variable names, but they match the math in Durbin and Koopman.
    const Selector &observed(observed_status(t));
    const DiagonalMatrix H = observation_variance(t, observed);
    const Vector &v(marg.prediction_error());

    Ptr<SparseKalmanMatrix> Finv = marg.sparse_forecast_precision();
    Ptr<SparseMatrixProduct> K(marg.sparse_kalman_gain(observed, Finv));

    Vector observation_error_mean = H * (*Finv * v - *K * r);
    Vector observation_error_variance =
        H.diag() - H * H * K->sparse_sandwich(N)->diag();
    // const double observation_error_mean = H * u;
    // const double observation_error_variance = H - H * D * H;
    // if (save_state_distributions) {
    //   marg.set_prediction_error(observation_error_mean);
    //   marg.set_prediction_variance(observation_error_variance);
    // }
    if (update_sufficient_statistics) {
       update_observation_model_complete_data_sufficient_statistics(
           t, observation_error_mean, observation_error_variance);
    }

    report_error("update_observation_model is not fully implemented.");
    if (gradient) {
      // update_observation_model_gradient(
      //      observation_parameter_component(*gradient), t, observation_error_mean,
      //      observation_error_variance);
    }

    // // Kalman smoother: convert r[t] to r[t-1] and N[t] to N[t-1].
    // sparse_scalar_kalman_disturbance_smoother_update(
    //     r, N, (*state_transition_matrix(t)), K, observation_matrix(t), F, v);
    report_error("CindBase::update_observation_model isn't done.");
  }

  Matrix MvBase::state_mean() const {
    const auto &kalman_filter(get_filter());
    Matrix ans(state_dimension(), time_dimension());
    for (size_t i = 0; i < time_dimension(); ++i) {
      ans.col(i) = kalman_filter[i].state_mean();
    }
    return ans;
  }

}  // namespace BOOM

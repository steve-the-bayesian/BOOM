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

#include "Models/StateSpace/StateSpaceModelBase.hpp"
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
    using Base = StateSpaceModelBase;
    using ScalarBase = ScalarStateSpaceModelBase;
  }  // namespace

  //----------------------------------------------------------------------
  Base::StateSpaceModelBase()
      : state_is_fixed_(false)
  {}

  Base::StateSpaceModelBase(const Base &rhs)
      : Model(rhs),
        state_is_fixed_(rhs.state_is_fixed_) {
    // Normally the parameter_positions_ vector starts off empty, and gets
    // modified by add_state.  However, if the vector is empty the first call to
    // add_state calls observation_model(), a virtual function, to get the size
    // of the observation model parameters.  We need to avoid that call because
    // virtual functions should not be called in constructors (and in this case
    // would give the wrong answer).  To get around the virtual function call,
    // we populate the first element of parameter_positions_ here.  Note that
    // this solution is tightly tied to the implementation of add_state, so if
    // that code changes in the future this constructor will probably need to
    // change as well.
    parameter_positions_.push_back(rhs.parameter_positions_[0]);
    for (int s = 0; s < rhs.number_of_state_models(); ++s) {
      add_state(rhs.state_model(s)->clone());
    }
    if (state_is_fixed_) state_ = rhs.state_;
  }

  Base &Base::operator=(const Base &rhs) {
    if (&rhs != this) {
      Model::operator=(rhs);
      state_models_.clear();
      state_is_fixed_ = rhs.state_is_fixed_;
      for (int s = 0; s < number_of_state_models(); ++s) {
        add_state(rhs.state_model(s)->clone());
      }
      if (state_is_fixed_) state_ = rhs.state_;
    }
    return *this;
  }

  // Copy the posterior samplers from rhs.
  void Base::copy_samplers(const Base &rhs) {
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
                state_model(s).get()));
      }
    }

    num_methods =rhs.number_of_sampling_methods();
    for (int m = 0; m < num_methods; ++m) {
      set_method(rhs.sampler(m)->clone_to_new_host(this));
    }
  }

  //----------------------------------------------------------------------
  namespace {
    void concatenate_parameter_vectors(std::vector<Ptr<Params>> &first,
                                       const std::vector<Ptr<Params>> &second) {
      std::copy(second.begin(), second.end(), std::back_inserter(first));
    }
  }  // namespace

  std::vector<Ptr<Params>> Base::parameter_vector() {
    std::vector<Ptr<Params>> ans;
    if (observation_model()) {
      concatenate_parameter_vectors(
          ans, observation_model()->parameter_vector());
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      concatenate_parameter_vectors(ans, state_model(s)->parameter_vector());
    }
    return ans;
  }

  const std::vector<Ptr<Params>> Base::parameter_vector() const {
    std::vector<Ptr<Params>> ans;
    if (observation_model()) {
      concatenate_parameter_vectors(
          ans, observation_model()->parameter_vector());
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      concatenate_parameter_vectors(ans, state_model(s)->parameter_vector());
    }
    return ans;
  }

  //----------------------------------------------------------------------
  VectorView Base::state_parameter_component(Vector &model_parameters,
                                             int s) const {
    int start = parameter_positions_[s];
    int size;
    if (s + 1 == number_of_state_models()) {
      size = model_parameters.size() - start;
    } else {
      size = parameter_positions_[s + 1] - start;
    }
    return VectorView(model_parameters, start, size);
  }

  ConstVectorView Base::state_parameter_component(
      const Vector &model_parameters, int s) const {
    int start = parameter_positions_[s];
    int size;
    if (s + 1 == number_of_state_models()) {
      size = model_parameters.size() - start;
    } else {
      size = parameter_positions_[s + 1] - start;
    }
    return ConstVectorView(model_parameters, start, size);
  }

  //----------------------------------------------------------------------
  VectorView Base::observation_parameter_component(
      Vector &model_parameters) const {
    if (parameter_positions_.empty()) {
      return VectorView(model_parameters);
    } else {
      int size = parameter_positions_[0];
      return VectorView(model_parameters, 0, size);
    }
  }

  ConstVectorView Base::observation_parameter_component(
      const Vector &model_parameters) const {
    if (parameter_positions_.empty()) {
      return ConstVectorView(model_parameters);
    } else {
      int size = parameter_positions_[0];
      return ConstVectorView(model_parameters, 0, size);
    }
  }
  //----------------------------------------------------------------------
  void Base::add_state(const Ptr<StateModel> &m) {
    state_models_.add_state(m);
    if (parameter_positions_.empty() && observation_model()) {
      // If no state has been added yet, add the size of the observation model
      // parameters, which come before the state model parameters in the
      // parameter vector.
      //
      // Also see the note in the copy constructor.  If this code changes, the
      // copy constructor will probably need to change as well.
      parameter_positions_.push_back(
          observation_model()->vectorize_params(true).size());
    }
    if (parameter_positions_.empty()) {
      parameter_positions_.push_back(m->vectorize_params(true).size());
    } else {
      parameter_positions_.push_back(parameter_positions_.back() +
                                     m->vectorize_params(true).size());
    }
  }

  //----------------------------------------------------------------------
  void Base::permanently_set_state(const Matrix &state) {
    if ((ncol(state) != time_dimension()) ||
        (nrow(state) != state_dimension())) {
      ostringstream err;
      err << "Wrong dimension of 'state' in "
          << "ScalarStateSpaceModelBase::permanently_set_state()."
          << "Argument was " << nrow(state) << " by " << ncol(state)
          << ".  Expected " << state_dimension() << " by " << time_dimension()
          << "." << endl;
      report_error(err.str());
    }
    state_is_fixed_ = true;
    state_ = state;
  }

  //----------------------------------------------------------------------
  void Base::observe_fixed_state() {
    clear_client_data();
    for (int t = 0; t < time_dimension(); ++t) {
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  //----------------------------------------------------------------------
  Vector Base::initial_state_mean() const {
    Vector ans;
    for (int s = 0; s < state_models_.size(); ++s) {
      ans.concat(state_model(s)->initial_state_mean());
    }
    return ans;
  }

  //----------------------------------------------------------------------
  SpdMatrix Base::initial_state_variance() const {
    // Ensure that the base state dimension is called.
    SpdMatrix ans(Base::state_dimension());
    int lo = 0;
    for (int s = 0; s < number_of_state_models(); ++s) {
      Ptr<StateModel> this_state_model = state_model(s);
      int hi = lo + this_state_model->state_dimension() - 1;
      SubMatrix block(ans, lo, hi, lo, hi);
      block = this_state_model->initial_state_variance();
      lo = hi + 1;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  void Base::clear_client_data() {
    if (observation_model()) {
      observation_model()->clear_data();
    }
    state_models_.clear_data();
    signal_complete_data_reset();
  }

  //----------------------------------------------------------------------
  void Base::register_data_observer(StateSpace::SufstatManagerBase *observer) {
    data_observers_.push_back(StateSpace::SufstatManager(observer));
  }

  //----------------------------------------------------------------------
  // Send a signal to any object observing this model's data that observation t
  // has changed.
  void Base::signal_complete_data_change(int t) {
    for (int i = 0; i < data_observers_.size(); ++i) {
      data_observers_[i].update_complete_data_sufficient_statistics(t);
    }
  }

  //----------------------------------------------------------------------
  void Base::set_state_model_behavior(StateModelBase::Behavior behavior) {
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->set_behavior(behavior);
    }
  }

  //----------------------------------------------------------------------
  void Base::impute_state(RNG &rng) {
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
      propagate_disturbances();
    }
  }

  namespace {
    // A functor that evaluates the log likelihood of a StateSpaceModelBase.
    // Suitable for passing to numerical optimizers.
    class StateSpaceTargetFun {
     public:
      explicit StateSpaceTargetFun(StateSpaceModelBase *model)
          : model_(model) {}

      double operator()(const Vector &parameters) {
        Vector old_parameters = model_->vectorize_params();
        model_->unvectorize_params(parameters);
        double ans = model_->log_likelihood();
        model_->unvectorize_params(old_parameters);
        return ans;
      }

     private:
      StateSpaceModelBase *model_;
    };
  }  // namespace

  //----------------------------------------------------------------------
  double Base::mle(double epsilon) {
    // If the model can be estimated using an EM algorithm, then do a
    // few steps of EM, and then switch to BFGS.
    Vector original_parameters = vectorize_params(true);
    if (check_that_em_is_legal()) {
      clear_client_data();
      double old_loglikelihood = Estep(false);
      double crit = 1 + epsilon;
      while (crit > std::min<double>(1.0, 100 * epsilon)) {
        Mstep(epsilon);
        clear_client_data();
        double log_likelihood = Estep(false);
        crit = log_likelihood - old_loglikelihood;
        old_loglikelihood = log_likelihood;
      }
    }

    StateSpaceTargetFun target(this);
    Negate min_target(target);
    PowellMinimizer minimizer(min_target);
    minimizer.set_evaluation_limit(500);
    Vector parameters = vectorize_params(true);
    if (parameters != original_parameters) {
      double stepsize = fabs(mean(parameters - original_parameters));
      minimizer.set_initial_stepsize(stepsize);
    }
    minimizer.set_precision(epsilon);
    minimizer.minimize(parameters);
    unvectorize_params(minimizer.minimizing_value());
    return log_likelihood();
  }

  //----------------------------------------------------------------------
  double Base::Estep(bool save_state_distributions) {
    return average_over_latent_data(true, save_state_distributions, nullptr);
  }

  //----------------------------------------------------------------------
  void Base::Mstep(double epsilon) {
    if (observation_model()) {
      observation_model()->find_posterior_mode(epsilon);
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->find_posterior_mode(epsilon);
    }
  }

  //----------------------------------------------------------------------
  bool Base::check_that_em_is_legal() const {
    if (observation_model()
        && !observation_model()->can_find_posterior_mode()) {
      return false;
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      if (!state_model(s)->can_find_posterior_mode()) {
        return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------
  Matrix Base::state_posterior_means() const {
    Matrix ans(state_dimension(), time_dimension());
    const KalmanFilterBase &filter(get_filter());
    for (int t = 0; t < time_dimension(); ++t) {
      ans.col(t) = filter[t].state_mean();
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Matrix Base::state_filtering_means() const {
    Matrix ans(state_dimension(), time_dimension());
    ans.col(0) = initial_state_mean();
    const KalmanFilterBase &filter(get_filter());
    for (int t = 1; t < time_dimension(); ++t) {
      ans.col(t) = filter[t - 1].state_mean();
    }
    return ans;
  }

  //----------------------------------------------------------------------
  const SpdMatrix &Base::state_posterior_variance(int t) const {
    return get_filter()[t].state_variance();
  }

  //----------------------------------------------------------------------
  double Base::log_likelihood() {
    return get_filter().compute_log_likelihood();
  }

  //----------------------------------------------------------------------
  double Base::log_likelihood(const Vector &parameters) {
    StateSpaceUtils::LogLikelihoodEvaluator evaluator(this);
    return evaluator.evaluate_log_likelihood(parameters);
  }

  //----------------------------------------------------------------------
  double Base::log_likelihood_derivatives(const Vector &parameters,
                                          Vector &gradient) {
    StateSpaceUtils::LogLikelihoodEvaluator evaluator(this);
    return evaluator.evaluate_log_likelihood_derivatives(
        ConstVectorView(parameters), VectorView(gradient));
  }

  //----------------------------------------------------------------------
  double Base::log_likelihood_derivatives(VectorView gradient) {
    Vector gradient_vector(gradient);
    double ans = average_over_latent_data(false, false, &gradient_vector);
    gradient = gradient_vector;
    return ans;
  }

  //----------------------------------------------------------------------
  void Base::simulate_initial_state(RNG &rng, VectorView state0) const {
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->simulate_initial_state(
          rng, state_component(state0, s));
    }
  }

  //----------------------------------------------------------------------
  // Simulates state for time period t
  void Base::simulate_next_state(RNG &rng, const ConstVectorView &last,
                                 VectorView next, int t) const {
    next = (*state_transition_matrix(t - 1)) * last
        + simulate_state_error(rng, t - 1);
  }

  //----------------------------------------------------------------------
  Vector Base::simulate_next_state(
      RNG &rng, const Vector &current_state, int t) const {
    Vector ans(current_state);
    simulate_next_state(rng, ConstVectorView(current_state),
                        VectorView(ans), t);
    return ans;
  }

  //----------------------------------------------------------------------
  void Base::advance_to_timestamp(RNG &rng, int &time, Vector &state,
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

  //----------------------------------------------------------------------
  Matrix Base::simulate_state_forecast(RNG &rng, int horizon) const {
    if (horizon < 0) {
      report_error(
          "simulate_state_forecast called with a negative "
          "forecast horizon.");
    }
    Matrix ans(state_dimension(), horizon + 1);
    int T = time_dimension();
    ans.col(0) = final_state();
    for (int i = 1; i <= horizon; ++i) {
      simulate_next_state(rng, ans.col(i - 1), ans.col(i), T + i);
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector Base::simulate_state_error(RNG &rng, int t) const {
    // simulate N(0, RQR) for the state at time t+1, using the
    // variance matrix at time t.
    Vector ans(state_dimension(), 0);
    for (int s = 0; s < number_of_state_models(); ++s) {
      VectorView eta(state_component(ans, s));
      state_model(s)->simulate_state_error(rng, eta, t);
    }
    return ans;
  }

  //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
  // Protected members
  //----------------------------------------------------------------------
  void Base::update_state_level_complete_data_sufficient_statistics(
      int t, const Vector &state_error_mean,
      const SpdMatrix &state_error_variance) {
    if (t >= 0) {
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->update_complete_data_sufficient_statistics(
            t, const_state_error_component(state_error_mean, s),
            state_error_variance_component(state_error_variance, s));
      }
    }
  }

  //----------------------------------------------------------------------
  void Base::observe_state(int t) {
    if (t == 0) {
      observe_initial_state();
      return;
    }
    const ConstVectorView now(state().col(t));
    const ConstVectorView then(state().col(t - 1));
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->observe_state(state_component(then, s),
                                    state_component(now, s), t);
    }
  }

  //----------------------------------------------------------------------
  void Base::observe_initial_state() {
    for (int s = 0; s < number_of_state_models(); ++s) {
      ConstVectorView state(state_component(state_.col(0), s));
      state_model(s)->observe_initial_state(state);
    }
  }

  //----------------------------------------------------------------------
  void Base::update_state_model_gradient(
      Vector *gradient, int t, const Vector &state_error_mean,
      const SpdMatrix &state_error_variance) {
    if (t >= 0) {
      for (int s = 0; s < number_of_state_models(); ++s) {
        state_model(s)->increment_expected_gradient(
            state_parameter_component(*gradient, s), t,
            const_state_error_component(state_error_mean, s),
            state_error_variance_component(state_error_variance, s));
      }
    }
  }

  //----------------------------------------------------------------------------
  // The math here comes from Durbin and Koopman 7.3.3.  The derivative of the
  // observed data log likelihood equals the expected derivative of the observed
  // data log likelihood, which is closely related to the objective function in
  // the E-step of the EM algorithm.  The expectation involves the full Kalman
  // filter and the disturbance smoother.
  //
  // The return value is the observed data log likelihood at the current
  // parameter values.  If 'update_sufficient_statistics' is true then this
  // implements the Estep of the EM algorithm by setting the complete data
  // sufficient statistics of the observation and state models, so that their
  // MLE's or MAP's can be computed in the M-step.
  //
  // If gradient is not the nullptr then it is filled with the gradient of log
  // likelihood.
  double Base::average_over_latent_data(bool update_sufficient_statistics,
                                        bool save_state_distributions,
                                        Vector *gradient) {
    if (update_sufficient_statistics) {
      clear_client_data();
    }
    if (gradient) {
      *gradient = vectorize_params(true) * 0.0;
    }
    // Compute log likelihood (the return value) and fill the kalman filter with
    // current values.
    kalman_filter();

    // This is the disturbance smoother from Durbin and Koopman, equation
    // (4.69).
    Vector r(state_dimension(), 0.0);
    SpdMatrix N(state_dimension(), 0.0);
    for (int t = time_dimension() - 1; t >= 0; --t) {
      update_observation_model(r, N, t, save_state_distributions,
                               update_sufficient_statistics, gradient);

      // The E step contribution for the observation at time t involves the mean
      // and the variance of the state error from time t-1.
      //
      // The formula for the state error mean in Durbin and Koopman is
      // equation (4.41):   \hat \eta_t = Q_t R'_t r_t.
      //
      // state_error_mean is \hat eta[t-1]
      const Vector state_error_mean = (*state_error_variance(t - 1)) *
                                      state_error_expander(t - 1)->Tmult(r);

      // The formula for the state error variance in Durbin and Koopman is
      // equation (4.47):
      //
      // Var(\eta_t | Y) = Q - QR'NRQ  // all subscripted by _t
      //
      // state_error_posterior_variance is Var(\hat eta[t-1] | Y).
      SpdMatrix state_error_posterior_variance =
          state_error_expander(t - 1)->sandwich_transpose(N);  // transpose??
      state_error_variance(t - 1)->sandwich_inplace(
          state_error_posterior_variance);
      state_error_posterior_variance *= -1;
      state_error_variance(t - 1)->add_to(state_error_posterior_variance);

      if (update_sufficient_statistics) {
        update_state_level_complete_data_sufficient_statistics(
            t - 1, state_error_mean, state_error_posterior_variance);
      }

      if (gradient) {
        update_state_model_gradient(gradient, t - 1, state_error_mean,
                                    state_error_posterior_variance);
      }

      if (save_state_distributions) {
        // Now r is r_{t-1} and N is N_{t-1}.  From Durbin and Koopman (4.32)
        // E(alpha[t] | Y) = a[t] + P * r[t-1]
        // V(alpha[t] | Y) = P[t] - P[t] * N[t-1] * P[t]
        const SpdMatrix &P(get_filter()[t].state_variance());
        get_filter()[t].increment_state_mean(P * r);
        get_filter()[t].increment_state_variance(-1 * sandwich(P, N));
      }
    }
    // The kalman filter is not current because it contains smoothed values.
    double loglike = get_filter().log_likelihood();
    get_filter().set_status(KalmanFilterBase::NOT_CURRENT);
    return loglike;
  }

  //----------------------------------------------------------------------
  // Send a signal to any observers of this model's data that the
  // complete data sufficient statistics should be reset.
  void Base::signal_complete_data_reset() {
    for (int i = 0; i < data_observers_.size(); ++i) {
      data_observers_[i].clear_complete_data_sufficient_statistics();
    }
  }

  // Ensure that state_ is large enough to hold the results of
  // impute_state().
  void Base::resize_state() {
    if (nrow(state_) != state_dimension() || ncol(state_) != time_dimension()) {
      state_.resize(state_dimension(), time_dimension());
    }
    for (int s = 0; s < number_of_state_models(); ++s) {
      state_model(s)->observe_time_dimension(time_dimension());
    }
  }


  //===========================================================================
  ScalarBase::ScalarStateSpaceModelBase() :
      filter_(this), simulation_filter_(this) {}

  ScalarBase::ScalarStateSpaceModelBase(const ScalarBase &rhs):
      Base(rhs),
      filter_(this),
      simulation_filter_(this)
  {}

  SparseVector ScalarBase::observation_matrix(int t) const {
    SparseVector ans;
    for (int s = 0; s < number_of_state_models(); ++s) {
      ans.concatenate(state_model(s)->observation_matrix(t));
    }
    return ans;
  }
  //----------------------------------------------------------------------
  void ScalarBase::kalman_filter() {
    filter_.update();
  }

  //----------------------------------------------------------------------
  Vector ScalarBase::one_step_prediction_errors(bool standardize) {
    kalman_filter();
    int n = time_dimension();
    Vector errors(n);
    if (n == 0) return errors;
    for (int i = 0; i < n; ++i) {
      errors[i] = filter_.prediction_error(i, standardize);
    }
    return errors;
  }

  //----------------------------------------------------------------------
  std::vector<Vector> ScalarBase::state_contributions() const {
    std::vector<Vector> ans(number_of_state_models());
    for (int t = 0; t < time_dimension(); ++t) {
      for (int m = 0; m < number_of_state_models(); ++m) {
        ConstVectorView state(state_component(this->state().col(t), m));
        ans[m].push_back(state_model(m)->observation_matrix(t).dot(state));
      }
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector ScalarBase::state_contribution(int which_model) const {
    const Matrix &state(this->state());
    if (ncol(state) != time_dimension() || nrow(state) != state_dimension()) {
      ostringstream err;
      err << "state is the wrong size in "
          << "ScalarStateSpaceModelBase::state_contribution" << endl
          << "State contribution matrix has " << ncol(state) << " columns.  "
          << "Time dimension is " << time_dimension() << "." << endl
          << "State contribution matrix has " << nrow(state) << " rows.  "
          << "State dimension is " << state_dimension() << "." << endl;
      report_error(err.str());
    }
    Vector ans(time_dimension());
    for (int t = 0; t < time_dimension(); ++t) {
      ConstVectorView local_state(state_component(state.col(t), which_model));
      ans[t] = state_model(which_model)->observation_matrix(t).dot(local_state);
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector ScalarBase::regression_contribution() const { return Vector(); }

  //----------------------------------------------------------------------
  Vector ScalarBase::observation_error_means() const {
    Vector ans(time_dimension());
    for (int i = 0; i < time_dimension(); ++i) {
      ans[i] = filter_.prediction_error(i);
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector ScalarBase::observation_error_variances() const {
    Vector ans(time_dimension());
    for (int i = 0; i < time_dimension(); ++i) {
      ans[i] = filter_[i].prediction_variance();
    }
    return ans;
  }

  //----------------------------------------------------------------------
  ScalarKalmanFilter &ScalarBase::get_filter() {
    return filter_;
  }
  const ScalarKalmanFilter &ScalarBase::get_filter() const {
    return filter_;
  }

  ScalarKalmanFilter &ScalarBase::get_simulation_filter() {
    return simulation_filter_;
  }

  const ScalarKalmanFilter &ScalarBase::get_simulation_filter() const {
    return simulation_filter_;
  }

  //----------------------------------------------------------------------
  // Simulate alpha_+ and y_* = y - y_+.  While simulating y_*,
  // feed it into the light (no storage for P) Kalman filter.  The
  // simulated state is stored in state_.
  //
  // y_+ and alpha_+ will be simulated in parallel with
  // Kalman filtering and disturbance smoothing of y, and the results
  // will be subtracted to compute y_*.
  void ScalarBase::simulate_forward(RNG &rng) {
    ScalarKalmanFilter &filter(get_filter());
    filter.update();
    ScalarKalmanFilter &simulation_filter(get_simulation_filter());
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
      simulation_filter.update(
          simulate_adjusted_observation(rng, t),
          t, is_missing_observation(t));
    }
  }

  //----------------------------------------------------------------------
  void ScalarBase::update_observation_model_complete_data_sufficient_statistics(
      int, double, double) {
    report_error(
        "To use an EM algorithm the model must override"
        " update_observation_model_complete_data_sufficient"
        "_statistics.");
  }
  //----------------------------------------------------------------------
  void ScalarBase::update_observation_model_gradient(VectorView, int, double,
                                                double) {
    report_error(
        "To numerically maximize the log likelihood or log "
        "posterior, the model must override "
        "update_observation_model_gradient.");
  }

  void ScalarBase::update_observation_model(Vector &r, SpdMatrix &N, int t,
                                       bool save_state_distributions,
                                       bool update_sufficient_statistics,
                                       Vector *gradient) {
    // Some syntactic sugar to make later formulas easier to read.  These are
    // bad variable names, but they match the math in Durbin and Koopman.
    const double H = observation_variance(t);
    Kalman::ScalarMarginalDistribution &marg(get_filter()[t]);

    const double F = marg.prediction_variance();
    const double v = marg.prediction_error();
    const Vector &K(marg.kalman_gain());

    double u = v / F - K.dot(r);
    double D = (1.0 / F) + N.Mdist(K);

    const double observation_error_mean = H * u;
    const double observation_error_variance = H - H * D * H;
    if (save_state_distributions) {
      marg.set_prediction_error(observation_error_mean);
      marg.set_prediction_variance(observation_error_variance);
    }
    if (update_sufficient_statistics) {
      update_observation_model_complete_data_sufficient_statistics(
          t, observation_error_mean, observation_error_variance);
    }
    if (gradient) {
      update_observation_model_gradient(
          observation_parameter_component(*gradient), t, observation_error_mean,
          observation_error_variance);
    }

    // Kalman smoother: convert r[t] to r[t-1] and N[t] to N[t-1].
    sparse_scalar_kalman_disturbance_smoother_update(
        r, N, (*state_transition_matrix(t)), K, observation_matrix(t), F, v);
  }

  //----------------------------------------------------------------------

  double ScalarBase::simulate_adjusted_observation(RNG &rng, int t) {
    double mu = observation_matrix(t).dot(state(t));
    return rnorm_mt(rng, mu, sqrt(observation_variance(t)));
  }

  //---------------------------------------------------------------------------
  // The call to simulate_forward fills the state matrix with simulated state
  // values that have the right variance but the wrong mean.  This function
  // subtracts off the wrong mean and adds in the correct one.
  void Base::propagate_disturbances() {
    if (time_dimension() <= 0) return;
    // Calling fast_disturbance_smoother() puts r[t] in
    // filter_[t].scaled_state_error().
    KalmanFilterBase &filter(get_filter());
    filter.fast_disturbance_smooth();
    KalmanFilterBase &simulation_filter(get_simulation_filter());
    simulation_filter.fast_disturbance_smooth();
    SpdMatrix P0 = initial_state_variance();

    // Propagates the r's forward to get E(alpha | y), and add it to the
    // simulated state.
    Vector state_mean_sim = initial_state_mean() +
        P0 * simulation_filter.initial_scaled_state_error();
    Vector state_mean_obs = initial_state_mean() +
        P0 * filter.initial_scaled_state_error();

    mutable_state().col(0) += state_mean_obs - state_mean_sim;
    observe_state(0);
    observe_data_given_state(0);

    for (int t = 1; t < time_dimension(); ++t) {
      state_mean_sim = (*state_transition_matrix(t - 1)) * state_mean_sim +
          (*state_variance_matrix(t - 1)) *
          simulation_filter[t - 1].scaled_state_error();
      state_mean_obs =
          (*state_transition_matrix(t - 1)) * state_mean_obs +
          (*state_variance_matrix(t - 1)) * filter[t - 1].scaled_state_error();

      mutable_state().col(t).axpy(state_mean_obs - state_mean_sim);
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  namespace StateSpaceUtils {

    // Compute one-step prediction errors on one or more holdout sets.
    std::vector<Matrix> compute_prediction_errors(
        const ScalarStateSpaceModelBase &model,
        int niter,
        const std::vector<int> &cutpoints,
        bool standardize) {
      std::vector<Matrix> prediction_errors(cutpoints.size(),
                                            Matrix(niter, model.time_dimension()));
      std::vector<std::future<void>> futures;
      int desired_threads = std::min<int>(
          cutpoints.size(),
          std::thread::hardware_concurrency() - 1);
      BOOM::ThreadWorkerPool pool;
      pool.add_threads(desired_threads);
      std::vector<Ptr<ScalarStateSpaceModelBase>> workers;

      class WorkWrapper {
       public:
        WorkWrapper(const Ptr<ScalarStateSpaceModelBase> worker,
                    int niter,
                    int cutpoint,
                    bool standardize,
                    Matrix &output)
            : worker_(worker),
              niter_(niter),
              cutpoint_(cutpoint),
              standardize_(standardize),
              output_(output)
        {}

        void operator()() {
          output_ = worker_->simulate_holdout_prediction_errors(
              niter_, cutpoint_, standardize_);
        }

       private:
        Ptr<ScalarStateSpaceModelBase> worker_;
        int niter_;
        int cutpoint_;
        bool standardize_;
        Matrix &output_;
      };

      for (int i = 0; i < cutpoints.size(); ++i) {
        workers.push_back(model.deepclone());
        futures.emplace_back(pool.submit(WorkWrapper(
            workers[i],
            niter,
            cutpoints[i],
            standardize,
            prediction_errors[i])));
      }
      for (int i = 0; i < futures.size(); ++i) {
        futures[i].get();
      }
      return prediction_errors;
    }
  }  // namespace StateSpaceUtils

}  // namespace BOOM

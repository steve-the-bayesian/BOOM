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
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "numopt.hpp"
#include "numopt/Powell.hpp"
#include "stats/moments.hpp"

namespace BOOM {
  namespace {
    using SSSMB = ScalarStateSpaceModelBase;
    using Base = StateSpaceModelBase;
  }  // namespace

  //----------------------------------------------------------------------
  Base::StateSpaceModelBase()
      : state_dimension_(0),
        state_error_dimension_(0),
        state_positions_(1, 0),
        state_error_positions_(1, 0),
        state_is_fixed_(false),
        kalman_filter_status_(NOT_CURRENT),
        log_likelihood_is_current_(false),
        state_transition_matrix_(new BlockDiagonalMatrix),
        state_variance_matrix_(new BlockDiagonalMatrix),
        state_error_expander_(new BlockDiagonalMatrix),
        state_error_variance_(new BlockDiagonalMatrix)
  {}

  Base::StateSpaceModelBase(const Base &rhs)
      : Model(rhs),
        state_dimension_(0),
        state_error_dimension_(0),
        state_positions_(1, 0),
        state_error_positions_(1, 0),
        state_is_fixed_(rhs.state_is_fixed_),
        kalman_filter_status_(rhs.kalman_filter_status_),
        log_likelihood_is_current_(rhs.log_likelihood_is_current_),
        state_transition_matrix_(new BlockDiagonalMatrix),
        state_variance_matrix_(new BlockDiagonalMatrix),
        state_error_expander_(new BlockDiagonalMatrix),
        state_error_variance_(new BlockDiagonalMatrix) {
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
    for (int s = 0; s < rhs.nstate(); ++s) {
      add_state(rhs.state_model(s)->clone());
    }
    if (state_is_fixed_) state_ = rhs.state_;
  }

  //----------------------------------------------------------------------
  void Base::add_state(const Ptr<StateModel> &m) {
    state_models_.push_back(m);
    state_dimension_ += m->state_dimension();
    int next_position = state_positions_.back() + m->state_dimension();
    state_positions_.push_back(next_position);

    state_error_dimension_ += m->state_error_dimension();
    next_position = state_error_positions_.back() + m->state_error_dimension();
    state_error_positions_.push_back(next_position);

    std::vector<Ptr<Params>> params(m->parameter_vector());
    for (int i = 0; i < params.size(); ++i) observe(params[i]);

    if (parameter_positions_.empty()) {
      // If no state has been added yet, add the size of the observation model
      // parameters, which come before the state model parameters in the
      // parameter vector.
      //
      // Also see the note in the copy constructor.  If this code changes, the
      // copy constructor will probably need to change as well.
      parameter_positions_.push_back(
          observation_model()->vectorize_params(true).size());
    }
    parameter_positions_.push_back(parameter_positions_.back() +
                                   m->vectorize_params(true).size());
  }

  //----------------------------------------------------------------------
  VectorView Base::state_component(Vector &state, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return VectorView(state, start, size);
  }

  //----------------------------------------------------------------------
  VectorView Base::state_component(VectorView &state, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return VectorView(state, start, size);
  }

  //----------------------------------------------------------------------
  ConstVectorView Base::state_component(const ConstVectorView &state,
                                        int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return ConstVectorView(state, start, size);
  }

  //----------------------------------------------------------------------
  ConstVectorView Base::const_state_error_component(const Vector &full_state_error,
                                                    int state_model_number) const {
    int start = state_error_positions_[state_model_number];
    int size = state_model(state_model_number)->state_error_dimension();
    return ConstVectorView(full_state_error, start, size);
  }

  VectorView Base::state_error_component(Vector &full_state_error,
                                         int state_model_number) const {
    int start = state_error_positions_[state_model_number];
    int size = state_model(state_model_number)->state_error_dimension();
    return VectorView(full_state_error, start, size);
  }

  //----------------------------------------------------------------------
  ConstSubMatrix Base::state_error_variance_component(
      const SpdMatrix &full_error_variance, int state) const {
    int start = state_error_positions_[state];
    int size = state_model(state)->state_error_dimension();
    return ConstSubMatrix(full_error_variance, start, start + size - 1, start,
                          start + size - 1);
  }

  //----------------------------------------------------------------------
  ConstSubMatrix Base::full_state_subcomponent(int state_model_index) const {
    int start = state_positions_[state_model_index];
    int size = state_model(state_model_index)->state_dimension();
    ConstSubMatrix contribution(state_, start, start + size - 1, 0,
                                time_dimension() - 1);
    return contribution;
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
    SpdMatrix ans(state_dimension_);
    int lo = 0;
    for (int s = 0; s < state_models_.size(); ++s) {
      Ptr<StateModel> this_state_model = state_model(s);
      int hi = lo + this_state_model->state_dimension() - 1;
      SubMatrix block(ans, lo, hi, lo, hi);
      block = this_state_model->initial_state_variance();
      lo = hi + 1;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  namespace {
    void concatenate_parameter_vectors(std::vector<Ptr<Params>> &first,
                                       const std::vector<Ptr<Params>> &second) {
      std::copy(second.begin(), second.end(), std::back_inserter(first));
    }
  }  // namespace

  ParamVector Base::parameter_vector() {
    std::vector<Ptr<Params>> ans;
    concatenate_parameter_vectors(ans, observation_model()->parameter_vector());
    for (int s = 0; s < nstate(); ++s) {
      concatenate_parameter_vectors(ans, state_model(s)->parameter_vector());
    }
    return ans;
  }

  const ParamVector Base::parameter_vector() const {
    std::vector<Ptr<Params>> ans;
    concatenate_parameter_vectors(ans, observation_model()->parameter_vector());
    for (int s = 0; s < nstate(); ++s) {
      concatenate_parameter_vectors(ans, state_model(s)->parameter_vector());
    }
    return ans;
  }

  //----------------------------------------------------------------------
  VectorView Base::state_parameter_component(Vector &model_parameters,
                                             int s) const {
    int start = parameter_positions_[s];
    int size;
    if (s + 1 == nstate()) {
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
    if (s + 1 == nstate()) {
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
  void Base::observe(const Ptr<Params> &parameter) {
    parameter->add_observer([this]() { this->kalman_filter_is_not_current(); });
  }

  //----------------------------------------------------------------------
  // TODO: This and other code involving model matrices is an optimization
  // opportunity.  Test it out to see if precomputation makes sense.
  const SparseKalmanMatrix *Base::state_transition_matrix(int t) const {
    // Size comparisons should be made with respect to state_dimension_, not
    // state_dimension() which is virtual.
    if (state_transition_matrix_->nrow() != state_dimension_ ||
        state_transition_matrix_->ncol() != state_dimension_) {
      state_transition_matrix_->clear();
      for (int s = 0; s < state_models_.size(); ++s) {
        state_transition_matrix_->add_block(
            state_model(s)->state_transition_matrix(t));
      }
    } else {
      // If we're in this block, then the matrix must have been created already,
      // and we just need to update the blocks.
      for (int s = 0; s < state_models_.size(); ++s) {
        state_transition_matrix_->replace_block(
            s, state_model(s)->state_transition_matrix(t));
      }
    }
    return state_transition_matrix_.get();
  }

  //----------------------------------------------------------------------
  const SparseKalmanMatrix *Base::state_variance_matrix(int t) const {
    state_variance_matrix_->clear();
    for (int s = 0; s < state_models_.size(); ++s) {
      state_variance_matrix_->add_block(
          state_model(s)->state_variance_matrix(t));
    }
    return state_variance_matrix_.get();
  }

  //----------------------------------------------------------------------
  const SparseKalmanMatrix *Base::state_error_expander(int t) const {
    state_error_expander_->clear();
    for (int s = 0; s < state_models_.size(); ++s) {
      state_error_expander_->add_block(state_model(s)->state_error_expander(t));
    }
    return state_error_expander_.get();
  }

  //----------------------------------------------------------------------
  const SparseKalmanMatrix *Base::state_error_variance(int t) const {
    state_error_variance_->clear();
    for (int s = 0; s < state_models_.size(); ++s) {
      state_error_variance_->add_block(state_model(s)->state_error_variance(t));
    }
    return state_error_variance_.get();
  }

  //----------------------------------------------------------------------
  void Base::clear_client_data() {
    observation_model()->clear_data();
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->clear_data();
    }
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
  void Base::set_state_model_behavior(StateModel::Behavior behavior) {
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->set_behavior(behavior);
    }
  }

  //----------------------------------------------------------------------
  void Base::impute_state(RNG &rng) {
    if (nstate() == 0) {
      report_error("No state has been defined.");
    }
    set_state_model_behavior(StateModel::MIXTURE);
    if (state_is_fixed_) {
      observe_fixed_state();
    } else {
      resize_state();
      clear_client_data();
      simulate_disturbances(rng);
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
    observation_model()->find_posterior_mode(epsilon);
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->find_posterior_mode(epsilon);
    }
  }

  //----------------------------------------------------------------------
  bool Base::check_that_em_is_legal() const {
    if (!observation_model()->can_find_posterior_mode()) return false;
    for (int s = 0; s < nstate(); ++s) {
      if (!state_model(s)->can_find_posterior_mode()) {
        return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------
  Matrix Base::state_posterior_means() const {
    Matrix ans(state_dimension(), time_dimension());
    for (int t = 0; t < time_dimension(); ++t) {
      ans.col(t) = kalman_state_storage(t).a;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Matrix Base::state_filtering_means() const {
    Matrix ans(state_dimension(), time_dimension());
    ans.col(0) = initial_state_mean();
    for (int t = 1; t < time_dimension(); ++t) {
      ans.col(t) = kalman_state_storage(t - 1).a;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  const SpdMatrix &Base::state_posterior_variance(int t) const {
    return kalman_state_storage(t).P;
  }

  //----------------------------------------------------------------------
  double Base::log_likelihood() {
    if (!log_likelihood_is_current_) {
      light_kalman_filter();
    }
    return log_likelihood_;
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
    for (int s = 0; s < state_models_.size(); ++s) {
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
      ostringstream err;
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
    Vector ans(state_dimension_, 0);
    for (int s = 0; s < state_models_.size(); ++s) {
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
      for (int s = 0; s < nstate(); ++s) {
        state_model(s)->update_complete_data_sufficient_statistics(
            t, const_state_error_component(state_error_mean, s),
            state_error_variance_component(state_error_variance, s));
      }
    }
  }

  //----------------------------------------------------------------------
  void ScalarStateSpaceModelBase::observe_state(int t) {
    if (t == 0) {
      observe_initial_state();
      return;
    }
    const ConstVectorView now(state().col(t));
    const ConstVectorView then(state().col(t - 1));
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->observe_state(state_component(then, s),
                                    state_component(now, s), t, this);
    }
  }

  //----------------------------------------------------------------------
  void Base::observe_initial_state() {
    for (int s = 0; s < nstate(); ++s) {
      ConstVectorView state(state_component(state_.col(0), s));
      state_model(s)->observe_initial_state(state);
    }
  }

  //----------------------------------------------------------------------
  void Base::update_state_model_gradient(
      Vector *gradient, int t, const Vector &state_error_mean,
      const SpdMatrix &state_error_variance) {
    if (t >= 0) {
      for (int s = 0; s < nstate(); ++s) {
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
    // The call to full_kalman_filter() does two things.  First, it
    // sets log_likelihood_, which is the return value. Second, it
    // fills kalman_storage_ with filtered values.
    full_kalman_filter();

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
        SpdMatrix &P(kalman_state_storage(t).P);
        kalman_state_storage(t).a += (P * r);
        P -= sandwich(P, N);
      }
    }
    // The kalman filter is not current because it contains smoothed values.
    set_kalman_filter_status(NOT_CURRENT);
    return log_likelihood();
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
    for (int s = 0; s < state_models_.size(); ++s) {
      state_models_[s]->observe_time_dimension(time_dimension());
    }
  }

  // Call each of the model matrices once to make sure any parameter observer
  // updates get done.
  void Base::update_model_matrices() {
    state_error_variance(0);
    state_variance_matrix(0);
    state_error_expander(0);
    state_transition_matrix(0);
  }
  //===========================================================================

  SparseVector SSSMB::observation_matrix(int t) const {
    SparseVector ans;
    for (int s = 0; s < nstate(); ++s) {
      ans.concatenate(state_model(s)->observation_matrix(t));
    }
    return ans;
  }
  //----------------------------------------------------------------------
  void SSSMB::kalman_filter(bool save_state_moments) {
    if (kalman_filter_status() == MCMC_CURRENT && log_likelihood_is_current()) {
      return;
    }
    check_kalman_storage(kalman_storage_, save_state_moments);
    double log_likelihood = 0;
    Vector conditional_state_mean = initial_state_mean();
    SpdMatrix conditional_state_variance = initial_state_variance();
    if (time_dimension() > 0) {
      for (int t = 0; t < time_dimension(); ++t) {
        if (save_state_moments) {
          kalman_storage_[t].a = conditional_state_mean;
          kalman_storage_[t].P = conditional_state_variance;
        }

        log_likelihood += sparse_scalar_kalman_update(
            adjusted_observation(t), conditional_state_mean,
            conditional_state_variance, kalman_storage_[t].K,
            kalman_storage_[t].F, kalman_storage_[t].v,
            is_missing_observation(t), observation_matrix(t),
            observation_variance(t), *state_transition_matrix(t),
            *state_variance_matrix(t));
        if (!std::isfinite(log_likelihood)) {
          set_log_likelihood(log_likelihood);
          set_kalman_filter_status(NOT_CURRENT);
          return;
        }
      }
    }
    if (!save_state_moments) {
      // We still record the final state mean and variance.
      kalman_storage_.back().a = conditional_state_mean;
      kalman_storage_.back().P = conditional_state_variance;
    }
    set_kalman_filter_status(MCMC_CURRENT);
    set_log_likelihood(log_likelihood);
  }
  //----------------------------------------------------------------------
  Vector SSSMB::one_step_prediction_errors() {
    int n = time_dimension();
    Vector errors(n);
    if (n == 0) return errors;
    if (kalman_filter_status() == NOT_CURRENT) {
      light_kalman_filter();
    }
    for (int i = 0; i < n; ++i) {
      errors[i] = kalman_storage_[i].v;
    }
    return errors;
  }

  //----------------------------------------------------------------------
  std::vector<Vector> SSSMB::state_contributions() const {
    std::vector<Vector> ans(nstate());
    for (int t = 0; t < time_dimension(); ++t) {
      for (int m = 0; m < nstate(); ++m) {
        ConstVectorView state(state_component(this->state().col(t), m));
        ans[m].push_back(state_model(m)->observation_matrix(t).dot(state));
      }
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector SSSMB::state_contribution(int which_model) const {
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
  Vector SSSMB::regression_contribution() const { return Vector(); }

  //----------------------------------------------------------------------
  Vector SSSMB::observation_error_means() const {
    Vector ans(kalman_storage_.size());
    for (int i = 0; i < kalman_storage_.size(); ++i) {
      ans[i] = kalman_storage_[i].v;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector SSSMB::observation_error_variances() const {
    Vector ans(kalman_storage_.size());
    for (int i = 0; i < kalman_storage_.size(); ++i) {
      ans[i] = kalman_storage_[i].F;
    }
    return ans;
  }

  void SSSMB::update_model_matrices() {
    observation_matrix(0);
    observation_variance(0);
    Base::update_model_matrices();
  }

  //----------------------------------------------------------------------
  // Simulate alpha_+ and y_* = y - y_+.  While simulating y_*,
  // feed it into the light (no storage for P) Kalman filter.  The
  // simulated state is stored in state_, while kalman_storage_ holds
  // the output of the Kalman filter.
  //
  // y_+ and alpha_+ will be simulated in parallel with
  // Kalman filtering and disturbance smoothing of y, and the results
  // will be subtracted to compute y_*.
  void SSSMB::simulate_forward(RNG &rng) {
    check_kalman_storage(kalman_storage_, false);
    check_kalman_storage(simulation_kalman_storage_, false);
    double log_likelihood = 0;
    Vector observed_data_state_mean = initial_state_mean();
    SpdMatrix observed_data_state_variance = initial_state_variance();

    Vector simulated_data_state_mean = observed_data_state_mean;
    SpdMatrix simulated_data_state_variance = observed_data_state_variance;
    for (int t = 0; t < time_dimension(); ++t) {
      // simulate_state at time t
      if (t == 0) {
        simulate_initial_state(rng, mutable_state().col(0));
      } else {
        simulate_next_state(rng, mutable_state().col(t - 1),
                            mutable_state().col(t), t);
      }
      sparse_scalar_kalman_update(
          simulate_adjusted_observation(rng, t),
          simulated_data_state_mean,
          simulated_data_state_variance,
          simulation_kalman_storage_[t].K,
          simulation_kalman_storage_[t].F,
          simulation_kalman_storage_[t].v,
          is_missing_observation(t),
          observation_matrix(t),
          observation_variance(t),
          *state_transition_matrix(t),
          *state_variance_matrix(t));
      log_likelihood += sparse_scalar_kalman_update(
          adjusted_observation(t),
          observed_data_state_mean,
          observed_data_state_variance,
          kalman_storage_[t].K,
          kalman_storage_[t].F,
          kalman_storage_[t].v,
          is_missing_observation(t),
          observation_matrix(t),
          observation_variance(t),
          (*state_transition_matrix(t)),
          (*state_variance_matrix(t)));
    }
    set_kalman_filter_status(MCMC_CURRENT);
    set_log_likelihood(log_likelihood);
  }

  void SSSMB::simulate_forward_and_filter(RNG &rng) {
    check_kalman_storage(simulation_kalman_storage_, false);
    Vector conditional_state_mean = initial_state_mean();
    SpdMatrix conditional_state_variance = initial_state_variance();
    for (int t = 0; t < time_dimension(); ++t) {
      if (t == 0) {
        simulate_initial_state(rng, mutable_state().col(0));
      } else {
        simulate_next_state(rng, mutable_state().col(t - 1),
                            mutable_state().col(t), t);
      }
      double y_sim = simulate_adjusted_observation(rng, t);
      sparse_scalar_kalman_update(
          y_sim, conditional_state_mean, conditional_state_variance,
          simulation_kalman_storage_[t].K,
          simulation_kalman_storage_[t].F,
          simulation_kalman_storage_[t].v, is_missing_observation(t),
          observation_matrix(t), observation_variance(t),
          (*state_transition_matrix(t)),
          (*state_variance_matrix(t)));
    }
  }

  void StateSpaceModelBase::simulate_disturbances(RNG &rng) {
    simulate_forward(rng);
    smooth_simulated_disturbances();
    smooth_observed_disturbances();
  }

  //----------------------------------------------------------------------
  void SSSMB::update_observation_model_complete_data_sufficient_statistics(
      int, double, double) {
    report_error(
        "To use an EM algorithm the model must override"
        " update_observation_model_complete_data_sufficient"
        "_statistics.");
  }
  //----------------------------------------------------------------------
  void SSSMB::update_observation_model_gradient(VectorView, int, double,
                                                double) {
    report_error(
        "To numerically maximize the log likelihood or log "
        "posterior, the model must override "
        "update_observation_model_gradient.");
  }

  void SSSMB::update_observation_model(Vector &r, SpdMatrix &N, int t,
                                       bool save_state_distributions,
                                       bool update_sufficient_statistics,
                                       Vector *gradient) {
    // Some syntactic sugar to make later formulas easier to read.  These are
    // bad variable names, but they match the math in Durbin and Koopman.
    const double H = observation_variance(t);
    const double F = kalman_storage_[t].F;
    const double v = kalman_storage_[t].v;
    const Vector &K(kalman_storage_[t].K);

    double u = v / F - K.dot(r);
    double D = (1.0 / F) + N.Mdist(K);

    const double observation_error_mean = H * u;
    const double observation_error_variance = H - H * D * H;
    if (save_state_distributions) {
      kalman_storage_[t].v = observation_error_mean;
      kalman_storage_[t].F = observation_error_variance;
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
  void SSSMB::check_kalman_storage(
      std::vector<ScalarKalmanStorage> &kalman_storage,
      bool save_state_moments) {
    // Set a flag to be flipped if any of the checks fail.
    bool ok = true;
    if (kalman_storage.size() < time_dimension()) {
      ok = false;
    }

    if (!kalman_storage.empty()) {
      if (kalman_storage[0].K.size() != state_dimension()) {
        ok = false;
      }
      if (save_state_moments) {
        if (kalman_storage[0].a.size() != state_dimension()) {
          ok = false;
        }
      }
    }

    // If any of the checks failed, blow kalman_storage away and refill with the
    // right size elements.
    if (!ok) {
      kalman_storage.assign(
          time_dimension(),
          ScalarKalmanStorage(state_dimension(), save_state_moments));
    }
  }

  //----------------------------------------------------------------------
  double SSSMB::simulate_adjusted_observation(RNG &rng, int t) {
    double mu = observation_matrix(t).dot(state(t));
    return rnorm_mt(rng, mu, sqrt(observation_variance(t)));
  }

  //----------------------------------------------------------------------
  // Disturbance smoother replaces Durbin and Koopman's K[t] with r[t].  The
  // disturbance smoother is equation (5) in Durbin and Koopman (2002).
  Vector SSSMB::smooth_disturbances_fast(
      std::vector<ScalarKalmanStorage> &filter) {
    int n = time_dimension();
    Vector r(state_dimension(), 0.0);
    for (int t = n - 1; t >= 0; --t) {
      // Upon entry r is r[t].
      // On exit, r is r[t-1] and filter[t].K is r[t]

      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z[t] * v[t]/F[t] + (T[t]^T - Z[t] * K[t]^T)r[t]
      //        = T[t]^T * r + Z[t] * (v[t]/F[t] - K.dot(r))

      // Some syntactic sugar makes the formulas easier to match up
      // with Durbin and Koopman.
      double v = filter[t].v;
      double F = filter[t].F;
      double coefficient = (v / F) - filter[t].K.dot(r);

      // Now produce r[t-1]
      Vector rt_1 = state_transition_matrix(t)->Tmult(r);
      observation_matrix(t).add_this_to(rt_1, coefficient);
      filter[t].K = r;
      r = rt_1;
    }
    return r;
  }

  //----------------------------------------------------------------------
  // After a call to smooth_disturbances_fast() puts r[t] in
  // kalman_storage_[t].K, this function propagates the r's forward to get
  // E(alpha | y), and add it to the simulated state.
  void SSSMB::propagate_disturbances() {
    if (time_dimension() <= 0) return;
    SpdMatrix P0 = initial_state_variance();
    Vector state_mean_sim = initial_state_mean() + P0 * r0_sim_;
    Vector state_mean_obs = initial_state_mean() + P0 * r0_obs_;

    mutable_state().col(0) += state_mean_obs - state_mean_sim;
    observe_state(0);
    observe_data_given_state(0);

    for (int t = 1; t < time_dimension(); ++t) {
      state_mean_sim = (*state_transition_matrix(t - 1)) * state_mean_sim +
                       (*state_variance_matrix(t - 1)) *
                           simulation_kalman_storage_[t - 1].K;
      state_mean_obs =
          (*state_transition_matrix(t - 1)) * state_mean_obs +
          (*state_variance_matrix(t - 1)) * kalman_storage_[t - 1].K;

      mutable_state().col(t).axpy(state_mean_obs - state_mean_sim);
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  //===========================================================================
  void MultivariateStateSpaceModelBase::kalman_filter(bool save_state_moments) {
    check_kalman_storage(kalman_storage_, save_state_moments);
    Vector conditional_state_mean = initial_state_mean();
    SpdMatrix conditional_state_variance = initial_state_variance();
    double log_likelihood = 0;
    for (int t = 0; t < time_dimension(); ++t) {
      if (save_state_moments) {
        kalman_storage_[t].a = conditional_state_mean;
        kalman_storage_[t].P = conditional_state_variance;
      }

      log_likelihood += sparse_multivariate_kalman_update(
          observation(t), conditional_state_mean, conditional_state_variance,
          kalman_storage_[t].kalman_gain_,
          kalman_storage_[t].forecast_precision_,
          kalman_storage_[t].forecast_precision_log_determinant_,
          kalman_storage_[t].forecast_error_, is_missing_observation(t),
          *observation_coefficients(t), observation_variance(t),
          *state_transition_matrix(t), *state_variance_matrix(t));
    }
    if (!save_state_moments) {
      // We still record the final state mean and variance.
      kalman_storage_.back().a = conditional_state_mean;
      kalman_storage_.back().P = conditional_state_variance;
    }
    set_kalman_filter_status(MCMC_CURRENT);
    set_log_likelihood(log_likelihood);
  }

  void MultivariateStateSpaceModelBase::update_model_matrices() {
    observation_variance(0);
    observation_coefficients(0);
    Base::update_model_matrices();
  }
  //---------------------------------------------------------------------------
  void MultivariateStateSpaceModelBase::simulate_forward(RNG &rng) {
    simulate_forward_and_filter(rng);
    light_kalman_filter();
  }

  void MultivariateStateSpaceModelBase::simulate_forward_and_filter(RNG &rng) {
    check_kalman_storage(simulation_kalman_storage_, false);
    Vector conditional_state_mean = initial_state_mean();
    SpdMatrix conditional_state_variance = initial_state_variance();
    for (int t = 0; t < time_dimension(); ++t) {
      if (t == 0) {
        simulate_initial_state(rng, mutable_state().col(0));
      } else {
        simulate_next_state(rng, mutable_state().col(t - 1),
                            mutable_state().col(t), t);
      }
      sparse_multivariate_kalman_update(
          simulate_observation(rng, t), conditional_state_mean,
          conditional_state_variance,
          simulation_kalman_storage_[t].kalman_gain_,
          simulation_kalman_storage_[t].forecast_precision_,
          simulation_kalman_storage_[t].forecast_precision_log_determinant_,
          simulation_kalman_storage_[t].forecast_error_,
          is_missing_observation(t), *observation_coefficients(t),
          observation_variance(t), *state_transition_matrix(t),
          *state_variance_matrix(t));
    }
  }

  //----------------------------------------------------------------------
  // Disturbance smoother replaces Durbin and Koopman's K[t] with r[t].  The
  // disturbance smoother is equation (5) in Durbin and Koopman (2002).
  Vector MultivariateStateSpaceModelBase::smooth_disturbances_fast(
      std::vector<MultivariateKalmanStorage> &filter) {
    int n = time_dimension();
    // The initial value of r, at time T is zero.
    Vector r(state_dimension(), 0.0);
    for (int t = n - 1; t >= 0; --t) {
      // Upon entry r is r[t].
      // On exit, r is r[t-1] and filter[t].r_ is r[t]

      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z[t]^T * (Finv[t] * v[t] - K[t]^T * r[t]) + T[t]^T * r[t]
      //
      // This formula is below formula 4.69 in Durbin and Koopman, second
      // edition, page 96.
      const SparseKalmanMatrix &Z(*observation_coefficients(t));
      Vector rt_1 =
          Z.Tmult(filter[t].forecast_precision_ * filter[t].forecast_error_ -
                  filter[t].kalman_gain_.Tmult(r)) +
          state_transition_matrix(t)->Tmult(r);
      filter[t].r_ = r;
      r = rt_1;
    }
    return r;
  }

  void MultivariateStateSpaceModelBase::update_observation_model(
      Vector &r, SpdMatrix &N, int t, bool save_state_distributions,
      bool update_sufficient_statistics, Vector *gradient) {
    SpdMatrix H = observation_variance(t);
    // u_t = F_t^{-1} v_t - K-t^T r_t
    Vector u = kalman_storage_[t].forecast_precision_ *
                   kalman_storage_[t].forecast_error_ -
               kalman_storage_[t].kalman_gain_.Tmult(r);

    // D_t = F_t^{-1} + K-t^T N_t K_t
    SpdMatrix D = kalman_storage_[t].forecast_precision_ +
                  sandwich_transpose(kalman_storage_[t].kalman_gain_, N);

    const Vector observation_error_mean = H * u;
    const SpdMatrix observation_error_variance = H - sandwich(H, D);
    if (save_state_distributions) {
      kalman_storage_[t].forecast_error_ = observation_error_mean;
      kalman_storage_[t].forecast_precision_ = observation_error_variance;
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
    sparse_multivariate_kalman_disturbance_smoother_update(
        r, N, (*state_transition_matrix(t)), kalman_storage_[t].kalman_gain_,
        *observation_coefficients(t), kalman_storage_[t].forecast_precision_,
        kalman_storage_[t].forecast_error_);
  }

  //----------------------------------------------------------------------
  // After a call to smooth_disturbances_fast() puts r[t] in
  // kalman_storage_[t].K, this function propagates the r's forward to get
  // E(alpha | y), and add it to the simulated state.
  void MultivariateStateSpaceModelBase::propagate_disturbances() {
    if (time_dimension() <= 0) return;
    SpdMatrix P0 = initial_state_variance();
    Vector state_mean_sim = initial_state_mean() + P0 * r0_sim_;
    Vector state_mean_obs = initial_state_mean() + P0 * r0_obs_;

    mutable_state().col(0) += state_mean_obs - state_mean_sim;
    observe_state(0);
    observe_data_given_state(0);

    for (int t = 1; t < time_dimension(); ++t) {
      state_mean_sim = (*state_transition_matrix(t - 1)) * state_mean_sim +
                       (*state_variance_matrix(t - 1)) *
                           simulation_kalman_storage_[t - 1].r_;
      state_mean_obs =
          (*state_transition_matrix(t - 1)) * state_mean_obs +
          (*state_variance_matrix(t - 1)) * kalman_storage_[t - 1].r_;
      mutable_state().col(t).axpy(state_mean_obs - state_mean_sim);
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  Vector MultivariateStateSpaceModelBase::simulate_observation(
      RNG &rng, int t) {
    return rmvn_mt(rng, *observation_coefficients(t) * state(t),
                   observation_variance(t));
  }

  void MultivariateStateSpaceModelBase::check_kalman_storage(
      std::vector<MultivariateKalmanStorage> &storage,
      bool save_state_moments) {
    if (time_dimension() == 0) {
      storage.clear();
      return;
    }
    if (storage.size() == time_dimension() + 1 &&
        (!save_state_moments ||
         ((storage[0].a.size() == state_dimension()) &&
          storage.back().a.size() == state_dimension()))) {
      return;
    }
    storage.resize(
        time_dimension(),
        MultivariateKalmanStorage(1, state_dimension(), save_state_moments));
  }

}  // namespace BOOM

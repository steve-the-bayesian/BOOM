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

#include <functional>
#include "distributions.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/StateSpace/Filters/SparseKalmanTools.hpp"
#include "cpputil/report_error.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "stats/moments.hpp"
#include "numopt.hpp"
#include "numopt/Powell.hpp"

namespace BOOM{
  namespace {
    typedef StateSpaceModelBase SSMB;
  }

  namespace StateSpace {
    MultiplexedData::MultiplexedData() : observed_sample_size_(0) {}

    // Child classes should call this function to make sure their missing status
    // and observed_sample_size_ are set correctly, but it does not actually
    // store data.
    void MultiplexedData::add_data(const Ptr<Data> &dp) {
      if (dp->missing() == Data::observed) {
        ++observed_sample_size_;
        if (this->missing() == Data::completely_missing) {
          set_missing_status(Data::partly_missing);
        }
      } else if (this->missing() == Data::observed) {
        if (observed_sample_size_ == 0) {
          set_missing_status(Data::completely_missing);
        } else {
          set_missing_status(Data::partly_missing);
        }
      }
    }

  }  // namespace StateSpace

  //----------------------------------------------------------------------
  SSMB::StateSpaceModelBase()
      : state_dimension_(0),
        state_error_dimension_(0),
        state_positions_(1, 0),
        state_error_positions_(1, 0),
        state_is_fixed_(false),
        mcmc_kalman_storage_is_current_(false),
        kalman_filter_is_current_(false),
        default_state_transition_matrix_(new BlockDiagonalMatrix),
        default_state_variance_matrix_(new BlockDiagonalMatrix),
        default_state_error_expander_(new BlockDiagonalMatrix),
        default_state_error_variance_(new BlockDiagonalMatrix)
  {}

  //----------------------------------------------------------------------
  SSMB::StateSpaceModelBase(const SSMB &rhs)
      : Model(rhs),
        state_dimension_(0),
        state_error_dimension_(0),
        state_positions_(1, 0),
        state_error_positions_(1, 0),
        state_is_fixed_(rhs.state_is_fixed_),
        mcmc_kalman_storage_is_current_(false),
        kalman_filter_is_current_(false),
        default_state_transition_matrix_(new BlockDiagonalMatrix),
        default_state_variance_matrix_(new BlockDiagonalMatrix),
        default_state_error_expander_(new BlockDiagonalMatrix),
        default_state_error_variance_(new BlockDiagonalMatrix)
  {
    // Normally the parameter_positions_ vector starts off empty, and
    // gets modified by add_state.  However, if the vector is empty
    // the first call to add_state calls observation_model() to get
    // the size of the observation model parameters.  We need to avoid
    // that call because virtual functions should be called in
    // constructors (and in this case would give the wrong answer).
    // To get around the virtual function call, we populate the first
    // element of parameter_positions_ here.  Note that this solution
    // is tightly tied to the implementation of add_state, so if that
    // code changes in the future this constructor will probably need
    // to change as well.
    parameter_positions_.push_back(rhs.parameter_positions_[0]);
    for (int s = 0; s < rhs.nstate(); ++s) {
      add_state(rhs.state_model(s)->clone());
    }
    if (state_is_fixed_) state_ = rhs.state_;
  }

  //----------------------------------------------------------------------
  namespace {
    void concatenate_parameter_vectors(std::vector<Ptr<Params>> &first,
                                       const std::vector<Ptr<Params>> &second) {
      std::copy(second.begin(), second.end(), std::back_inserter(first));
    }
  }  // namespace

  ParamVector SSMB::parameter_vector() {
    std::vector<Ptr<Params> > ans;
    concatenate_parameter_vectors(ans, observation_model()->parameter_vector());
    for (int s = 0; s < nstate(); ++s) {
      concatenate_parameter_vectors(ans, state_model(s)->parameter_vector());
    }
    return ans;
  }

  const ParamVector SSMB::parameter_vector() const {
    std::vector<Ptr<Params> > ans;
    concatenate_parameter_vectors(ans, observation_model()->parameter_vector());
    for (int s = 0; s < nstate(); ++s) {
      concatenate_parameter_vectors(ans, state_model(s)->parameter_vector());
    }
    return ans;
  }

  //----------------------------------------------------------------------
  int SSMB::state_dimension() const {return state_dimension_;}

  //----------------------------------------------------------------------
  void SSMB::impute_state(RNG &rng) {
    if (nstate() == 0) {
      report_error("No state has been defined.");
    }
    set_state_model_behavior(StateModel::MIXTURE);
    if (state_is_fixed_) {
      observe_fixed_state();
    } else {
      resize_state();
      clear_client_data();
      simulate_forward(rng);
      Vector r0_sim = smooth_disturbances_fast(light_kalman_storage_);
      Vector r0_obs = smooth_disturbances_fast(supplemental_kalman_storage_);
      propagate_disturbances(r0_sim, r0_obs, true);
    }
  }

  //----------------------------------------------------------------------
  // Ensure that state_ is large enough to hold the results of
  // impute_state().
  void SSMB::resize_state() {
    if (nrow(state_) != state_dimension()
       || ncol(state_) != time_dimension()) {
      state_.resize(state_dimension(), time_dimension());
    }
    for (int s = 0; s < state_models_.size(); ++s) {
      state_models_[s]->observe_time_dimension(time_dimension());
    }
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
  void SSMB::simulate_forward(RNG &rng) {
    check_light_kalman_storage(light_kalman_storage_);
    check_light_kalman_storage(supplemental_kalman_storage_);
    log_likelihood_ = 0;
    for (int t = 0; t < time_dimension(); ++t) {
      // simulate_state at time t
      if (t == 0) {
        simulate_initial_state(rng, state_.col(0));
        a_ = initial_state_mean();
        P_ = initial_state_variance();
        supplemental_a_ = a_;
        supplemental_P_ = P_;
      }else{
        simulate_next_state(rng, state_.col(t-1), state_.col(t), t);
      }
      double y_sim = simulate_adjusted_observation(rng, t);
      sparse_scalar_kalman_update(
          y_sim,
          a_,
          P_,
          light_kalman_storage_[t].K,
          light_kalman_storage_[t].F,
          light_kalman_storage_[t].v,
          is_missing_observation(t),
          observation_matrix(t),
          observation_variance(t),
          *state_transition_matrix(t),
          *state_variance_matrix(t));
        ////////////////////////
        // TODO(stevescott): The actual one step ahead prediction
        // errors are being stored in supplemental_kalman_storage_,
        // and not light_kalman_storage_.  We should eventually keep
        // the prediction errors in the right place.
      log_likelihood_ += sparse_scalar_kalman_update(
          adjusted_observation(t),
          supplemental_a_,
          supplemental_P_,
          supplemental_kalman_storage_[t].K,
          supplemental_kalman_storage_[t].F,
          supplemental_kalman_storage_[t].v,
          is_missing_observation(t),
          observation_matrix(t),
          observation_variance(t),
          (*state_transition_matrix(t)),
          (*state_variance_matrix(t)));

      // The Kalman update sets a_ to a[t+1] and P to P[t+1], so they
      // will be current for the next iteration.
    }
    mcmc_kalman_storage_is_current_ = true;
  }

  //----------------------------------------------------------------------
  double SSMB::simulate_adjusted_observation(RNG &rng, int t) {
    double mu = observation_matrix(t).dot(state_.col(t));
    return rnorm_mt(rng, mu, sqrt(observation_variance(t)));
  }

  //----------------------------------------------------------------------
  // Disturbance smoother replaces Durbin and Koopman's K[t] with
  // r[t].  The disturbance smoother is equation (5) in Durbin and
  // Koopman (2002).
  // TODO(stevescott): make sure you've got t, t-1, and t+1 worked out
  // correctly.
  Vector SSMB::smooth_disturbances_fast(
      std::vector<LightKalmanStorage> &kalman_storage) {
    int n = time_dimension();
    Vector r(state_dimension(), 0.0);
    for (int t = n-1; t>=0; --t) {
      // Upon entry r is r[t].
      // On exit, r is r[t-1] and kalman_storage[t].K is r[t]

      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z[t] * v[t]/F[t] + (T[t]^T - Z[t] * K[t]^T)r[t]
      //        = T[t]^T * r + Z[t] * (v[t]/F[t] - K.dot(r))

      // Some syntactic sugar makes the formulas easier to match up
      // with Durbin and Koopman.
      double v = kalman_storage[t].v;
      double F = kalman_storage[t].F;
      Vector &K(kalman_storage[t].K);
      double coefficient = (v/F) - K.dot(r);

      // Now produce r[t-1]
      Vector rt_1 = state_transition_matrix(t)->Tmult(r);
      observation_matrix(t).add_this_to(rt_1, coefficient);
      K = r;
      r = rt_1;
    }
    return r;
  }

  //----------------------------------------------------------------------
  // After a call to smooth_disturbances_fast() puts r[t] in
  // light_kalman_storage_[t].K, this function propagates the r's
  // forward to get E(alpha | y), and add it to the simulated state.
  void SSMB::propagate_disturbances(
      const Vector &r0_sim, const Vector & r0_obs, bool observe) {
    // TODO(stevescott): Two linear operations are being performed in
    // parallel.  Can they be replaced by a single linear operation on
    // the difference?
    if (state_.ncol() <= 0) return;
    SpdMatrix P0 = initial_state_variance();
    Vector state_mean_sim = initial_state_mean() + P0*r0_sim;
    Vector state_mean_obs = initial_state_mean() + P0*r0_obs;

    state_.col(0) += state_mean_obs - state_mean_sim;
    if (observe) {
      observe_state(0);
      observe_data_given_state(0);
    }
    for (int t = 1; t < time_dimension(); ++t) {
      state_mean_sim = (*state_transition_matrix(t-1)) * state_mean_sim
          + (*state_variance_matrix(t-1)) * light_kalman_storage_[t-1].K;
      state_mean_obs = (*state_transition_matrix(t-1)) * state_mean_obs
          + (*state_variance_matrix(t-1)) * supplemental_kalman_storage_[t-1].K;

      state_.col(t).axpy(state_mean_obs - state_mean_sim);
      if (observe) {
        observe_state(t);
        observe_data_given_state(t);
      }
    }
  }

  void SSMB::observe_state(int t) {
    if (t == 0) {
      observe_initial_state();
      return;
    }
    const ConstVectorView now(state_.col(t));
    const ConstVectorView then(state_.col(t-1));
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->observe_state(
          state_component(then, s),
          state_component(now, s),
          t);
    }
  }

  void SSMB::observe_initial_state() {
    for (int s = 0; s < nstate(); ++s) {
      ConstVectorView state(state_component(state_.col(0), s));
      state_model(s)->observe_initial_state(state);
    }
  }

  //----------------------------------------------------------------------
  Vector SSMB::one_step_prediction_errors() const {
    int n = time_dimension();
    Vector errors(n);
    if (n == 0) return errors;

    if (mcmc_kalman_storage_is_current_) {
      for (int i = 0; i < n; ++i) {
        // TODO(stevescott): Clean up this hack by making sure the one
        // step prediction errors are stored in light_kalman_storage_
        // instead of supplemental_kalman_storage_.
        errors[i] = supplemental_kalman_storage_[i].v;
      }
      return errors;
    }
    log_likelihood_ = 0;
    initialize_final_kalman_storage();
    ScalarKalmanStorage &ks(final_kalman_storage_);

    for (int i = 0; i < n; ++i) {
      double resid = adjusted_observation(i);
      bool missing = is_missing_observation(i);
      log_likelihood_ += sparse_scalar_kalman_update(
          resid,
          ks.a,
          ks.P,
          ks.K,
          ks.F,
          ks.v,
          missing,
          observation_matrix(i),
          observation_variance(i),
          (*state_transition_matrix(i)),
          (*state_variance_matrix(i)));
      errors[i] = ks.v;
    }
    kalman_filter_is_current_ = true;
    return errors;
  }
  //----------------------------------------------------------------------
  void SSMB::update_observation_model_complete_data_sufficient_statistics(
      int t,
      double observation_error_mean,
      double observation_error_variance) {
    report_error("To use an EM algorithm the model must override"
                 " update_observation_model_complete_data_sufficient"
                 "_statistics.");
  }
  //----------------------------------------------------------------------
  void SSMB::update_observation_model_gradient(
      VectorView gradient,
      int t,
      double observation_error_mean,
      double observation_error_variance) {
    report_error("To numerically maximize the log likelihood or log "
                 "posterior, the model must override "
                 "update_observation_model_gradient.");
  }
  //----------------------------------------------------------------------
  void SSMB::update_state_level_complete_data_sufficient_statistics(
      int t,
      const Vector &state_error_mean,
      const SpdMatrix &state_error_variance) {
    if (t >= 0) {
      for (int s = 0; s < nstate(); ++s) {
        state_model(s)->update_complete_data_sufficient_statistics(
            t,
            state_error_component(state_error_mean, s),
            state_error_variance_component(state_error_variance, s));
      }
    }
  }
  //----------------------------------------------------------------------
  void SSMB::update_state_model_gradient(
      Vector *gradient,
      int t,
      const Vector &state_error_mean,
      const SpdMatrix &state_error_variance) {
    if (t >= 0) {
      for (int s = 0; s < nstate(); ++s) {
        state_model(s)->increment_expected_gradient(
            state_parameter_component(*gradient, s),
            t,
            state_error_component(state_error_mean, s),
            state_error_variance_component(state_error_variance, s));
      }
    }
  }

  //----------------------------------------------------------------------
  void SSMB::clear_client_data() {
    observation_model()->clear_data();
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->clear_data();
    }
    signal_complete_data_reset();
  }

  //----------------------------------------------------------------------
  void SSMB::add_state(const Ptr<StateModel> & m) {
    state_models_.push_back(m);
    state_dimension_ += m->state_dimension();
    int next_position = state_positions_.back()
          + m->state_dimension();
    state_positions_.push_back(next_position);

    state_error_dimension_ += m->state_error_dimension();
    next_position = state_error_positions_.back() + m->state_error_dimension();
    state_error_positions_.push_back(next_position);

    std::vector<Ptr<Params> > params(m->parameter_vector());
    for (int i = 0; i < params.size(); ++i) observe(params[i]);

    if (parameter_positions_.empty()) {
      // See the note in the copy constructor.  If this code changes,
      // the copy constructor will probably need to change as well.
      parameter_positions_.push_back(
          observation_model()->vectorize_params(true).size());
    }
    parameter_positions_.push_back(
        parameter_positions_.back() + m->vectorize_params(true).size());
  }

  //----------------------------------------------------------------------
  SparseVector SSMB::observation_matrix(int t) const {
    SparseVector ans;
    for (int s = 0; s < nstate(); ++s) {
      ans.concatenate(state_models_[s]->observation_matrix(t));
    }
    return ans;
  }

  //----------------------------------------------------------------------
  // TODO(stevescott): This and other code involving model matrices is
  // an optimization opportunity.  Test it out to see if
  // precomputation makes sense.
  const SparseKalmanMatrix * SSMB::state_transition_matrix(int t) const {
    // Size comparisons should be made with respect to
    // state_dimension_, not state_dimension() which is virtual.
    if (default_state_transition_matrix_->nrow() != state_dimension_
       || default_state_transition_matrix_->ncol() != state_dimension_) {
      default_state_transition_matrix_->clear();
      for (int s = 0; s < state_models_.size(); ++s) {
        default_state_transition_matrix_->add_block(
            state_models_[s]->state_transition_matrix(t));
      }
    }else{
      // If we're in this block, then the matrix must have been
      // created already, and we just need to update the blocks.
      for (int s = 0; s < state_models_.size(); ++s) {
        default_state_transition_matrix_->replace_block(
            s, state_models_[s]->state_transition_matrix(t));
      }
    }
    return default_state_transition_matrix_.get();
  }

  //----------------------------------------------------------------------
  const SparseKalmanMatrix * SSMB::state_variance_matrix(int t) const {
    default_state_variance_matrix_->clear();
    for (int s = 0; s < state_models_.size(); ++s) {
      default_state_variance_matrix_->add_block(
          state_models_[s]->state_variance_matrix(t));
    }
    return default_state_variance_matrix_.get();
  }

  //----------------------------------------------------------------------
  const SparseKalmanMatrix * SSMB::state_error_expander(int t) const {
    default_state_error_expander_->clear();
    for (int s = 0; s < state_models_.size(); ++s) {
      default_state_error_expander_->add_block(
          state_models_[s]->state_error_expander(t));
    }
    return default_state_error_expander_.get();
  }

  //----------------------------------------------------------------------
  const SparseKalmanMatrix * SSMB::state_error_variance(int t) const {
    default_state_error_variance_->clear();
    for (int s = 0; s < state_models_.size(); ++s) {
      default_state_error_variance_->add_block(
          state_models_[s]->state_error_variance(t));
    }
    return default_state_error_variance_.get();
  }


  //----------------------------------------------------------------------
  double SSMB::log_likelihood() const {
    filter();
    return log_likelihood_;
  }

  double SSMB::log_likelihood(const Vector &parameters) const {
    StateSpaceUtils::LogLikelihoodEvaluator evaluator(this);
    return evaluator.evaluate_log_likelihood(parameters);
  }

  double SSMB::log_likelihood_derivatives(const Vector &parameters,
                                          Vector &gradient) const {
    StateSpaceUtils::LogLikelihoodEvaluator evaluator(this);
    return evaluator.evaluate_log_likelihood_derivatives(
        ConstVectorView(parameters),
        VectorView(gradient));
  }

  double SSMB::log_likelihood_derivatives(VectorView gradient) {
    Vector gradient_vector(gradient);
    double ans = average_over_latent_data(false, false, &gradient_vector);
    gradient = gradient_vector;
    return ans;
  }

  //----------------------------------------------------------------------
  const ScalarKalmanStorage & SSMB::filter() const {
    if (kalman_filter_is_current_) return final_kalman_storage_;
    initialize_final_kalman_storage();
    log_likelihood_ = 0;
    int n = time_dimension();
    if (n == 0) return final_kalman_storage_;
    ScalarKalmanStorage *ks = &final_kalman_storage_;
    for (int i = 0; i < n; ++i) {
      double resid = adjusted_observation(i);
      bool missing = is_missing_observation(i);
      log_likelihood_ += sparse_scalar_kalman_update(
          resid,
          ks->a,
          ks->P,
          ks->K,
          ks->F,
          ks->v,
          missing,
          observation_matrix(i),
          observation_variance(i),
          (*state_transition_matrix(i)),
          (*state_variance_matrix(i)));
    }
    kalman_filter_is_current_ = true;
    return final_kalman_storage_;
  }
  //----------------------------------------------------------------------
  const ScalarKalmanStorage & SSMB::full_kalman_filter() {
    allocate_full_kalman_filter(time_dimension(), state_dimension());
    log_likelihood_ = 0;
    initialize_final_kalman_storage();
    if (time_dimension() == 0) {
      return final_kalman_storage_;
    }
    full_kalman_storage_[0].a = final_kalman_storage_.a;
    full_kalman_storage_[0].P = final_kalman_storage_.P;
    for (int t = 0; t < time_dimension(); ++t) {
      full_kalman_storage_[t+1].a = full_kalman_storage_[t].a;
      full_kalman_storage_[t+1].P = full_kalman_storage_[t].P;
      double resid = adjusted_observation(t);
      log_likelihood_ += sparse_scalar_kalman_update(
          resid,
          full_kalman_storage_[t+1].a,
          full_kalman_storage_[t+1].P,
          full_kalman_storage_[t].K,
          full_kalman_storage_[t].F,
          full_kalman_storage_[t].v,
          is_missing_observation(t),
          observation_matrix(t),
          observation_variance(t),
          (*state_transition_matrix(t)),
          (*state_variance_matrix(t)));
    }
    kalman_filter_is_current_ = true;
    final_kalman_storage_ = full_kalman_storage_[time_dimension() - 1];
    return final_kalman_storage_;
  }
  //----------------------------------------------------------------------
  void SSMB::simulate_initial_state(RNG &rng, VectorView state0) const {
    for (int s = 0; s < state_models_.size(); ++s) {
      state_model(s)->simulate_initial_state(rng, state_component(state0, s));
    }
  }

  //----------------------------------------------------------------------
  // Simulates state for time period t
  void SSMB::simulate_next_state(RNG &rng,
                                 const ConstVectorView &last,
                                 VectorView next,
                                 int t) const {
    next= (*state_transition_matrix(t-1)) * last;
    next += simulate_state_error(rng, t-1);
  }

  //----------------------------------------------------------------------
  Vector SSMB::simulate_next_state(RNG &rng,
                                   const Vector &state,
                                   int t) const {
    Vector ans(state);
    simulate_next_state(rng,
                        ConstVectorView(state),
                        VectorView(ans),
                        t);
    return ans;
  }

  Matrix SSMB::simulate_state_forecast(RNG &rng, int horizon) const {
    if (horizon < 0) {
      report_error("simulate_state_forecast called with a negative "
                   "forecast horizon.");
    }
    Matrix ans(state_dimension(), horizon + 1);
    int T = time_dimension();
    ans.col(0) = final_state();
    for (int i = 1; i <= horizon; ++i) {
      simulate_next_state(rng, ans.col(i-1), ans.col(i), T + i);
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector SSMB::simulate_state_error(RNG &rng, int t) const {
    // simulate N(0, RQR) for the state at time t+1, using the
    // variance matrix at time t.
    Vector ans(state_dimension_, 0);
    for (int s = 0; s < state_models_.size(); ++s) {
      VectorView eta(state_component(ans, s));
      state_model(s)->simulate_state_error(rng, eta, t);
    }
    return ans;
  }
  //----------------------------------------------------------------------
  Vector SSMB::initial_state_mean() const {
    Vector ans;
    for (int s = 0; s < state_models_.size(); ++s) {
      ans.concat(state_models_[s]->initial_state_mean());
    }
    return ans;
  }

  //----------------------------------------------------------------------
  SpdMatrix SSMB::initial_state_variance() const {
    SpdMatrix ans(state_dimension_);
    int lo = 0;
    for (int s = 0; s < state_models_.size(); ++s) {
      Ptr<StateModel> state = state_models_[s];
      int hi = lo + state->state_dimension() - 1;
      SubMatrix block(ans, lo, hi, lo, hi);
      block = state_models_[s]->initial_state_variance();
      lo = hi + 1;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  void SSMB::observe(const Ptr<Params> & p) {
    p->add_observer(
        [this](){this->kalman_filter_is_not_current();});
  }

  //----------------------------------------------------------------------
  ConstVectorView SSMB::final_state() const {
    return state_.last_col();
  }

  //----------------------------------------------------------------------
  ConstVectorView SSMB::state(int t) const {
    return state_.col(t);
  }

  //----------------------------------------------------------------------
  const Matrix &SSMB::state() const {return state_;}

  //----------------------------------------------------------------------
  std::vector<Vector> SSMB::state_contributions() const {
    std::vector<Vector> ans(nstate());
    for (int t = 0; t < time_dimension(); ++t) {
      for (int m = 0; m < nstate(); ++m) {
        ConstVectorView state(state_component(state_.col(t), m));
        ans[m].push_back(state_models_[m]->observation_matrix(t).dot(state));
      }
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector SSMB::state_contribution(int s) const {
    if (ncol(state_) != time_dimension() ||
       nrow(state_) != state_dimension()) {
      ostringstream err;
      err << "state is the wrong size in "
          << "StateSpaceModelBase::state_contribution" << endl
          << "State contribution matrix has " << ncol(state_) << " columns.  "
          << "Time dimension is " << time_dimension() << "." << endl
          << "State contribution matrix has " << nrow(state_) << " rows.  "
          << "State dimension is " << state_dimension() << "." << endl;
      report_error(err.str());
    }
    Vector ans(time_dimension());
    for (int t = 0; t < time_dimension(); ++t) {
      ConstVectorView state(state_component(state_.col(t), s));
      ans[t] = state_model(s)->observation_matrix(t).dot(state);
    }
    return ans;
  }

  Vector SSMB::regression_contribution() const {
    return Vector();
  }

  //----------------------------------------------------------------------
  VectorView SSMB::state_component(Vector &v, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return VectorView(v, start, size);
  }

  //----------------------------------------------------------------------
  VectorView SSMB::state_component(VectorView &v, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return VectorView(v, start, size);
  }

  //----------------------------------------------------------------------
  ConstVectorView SSMB::state_component(const ConstVectorView &v, int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    return ConstVectorView(v, start, size);
  }

  //----------------------------------------------------------------------
  VectorView SSMB::observation_parameter_component(Vector &v) const {
    if (parameter_positions_.empty()) {
      return VectorView(v);
    } else {
      int size = parameter_positions_[0];
      return VectorView(v, 0, size);
    }
  }
  ConstVectorView SSMB::observation_parameter_component(const Vector &v) const {
    if (parameter_positions_.empty()) {
      return ConstVectorView(v);
    } else {
      int size = parameter_positions_[0];
      return ConstVectorView(v, 0, size);
    }
  }
  //----------------------------------------------------------------------
  VectorView SSMB::state_parameter_component(Vector &v, int s) const {
    int start = parameter_positions_[s];
    int size;
    if (s + 1 == nstate()) {
      size = v.size() - start;
    } else {
      size = parameter_positions_[s + 1] - start;
    }
    return VectorView(v, start, size);
  }
  ConstVectorView SSMB::state_parameter_component(const Vector &v, int s) const {
    int start = parameter_positions_[s];
    int size;
    if (s + 1 == nstate()) {
      size = v.size() - start;
    } else {
      size = parameter_positions_[s + 1] - start;
    }
    return ConstVectorView(v, start, size);
  }
  //----------------------------------------------------------------------
  ConstVectorView SSMB::state_error_component(
      const Vector &v, int s) const {
    int start = state_error_positions_[s];
    int size = state_model(s)->state_error_dimension();
    return ConstVectorView(v, start, size);
  }

  //----------------------------------------------------------------------
  ConstSubMatrix SSMB::state_error_variance_component(
      const SpdMatrix &full_error_variance,
      int s) const {
    int start = state_error_positions_[s];
    int size = state_model(s)->state_error_dimension();
    return ConstSubMatrix(full_error_variance,
                     start, start + size - 1,
                     start, start + size - 1);
  }

  //----------------------------------------------------------------------
  Matrix SSMB::full_state_subcomponent(int s) const {
    int start = state_positions_[s];
    int size = state_model(s)->state_dimension();
    ConstSubMatrix contribution(
        state_, start, start + size - 1, 0, time_dimension() - 1);
    return contribution.to_matrix();
  }

  //----------------------------------------------------------------------
  void SSMB::permanently_set_state(const Matrix &state) {
    if ((ncol(state) != time_dimension()) ||
       (nrow(state) != state_dimension())) {
      ostringstream err;
      err << "Wrong dimension of 'state' in "
          << "StateSpaceModelBase::permanently_set_state()."
          << "Argument was " << nrow(state) << " by " << ncol(state)
          << ".  Expected " << state_dimension() << " by "
          << time_dimension() << "." << endl;
      report_error(err.str());
    }
    state_is_fixed_ = true;
    state_ = state;
  }

  //----------------------------------------------------------------------
  void SSMB::observe_fixed_state() {
    clear_client_data();
    for (int t = 0; t < time_dimension(); ++t) {
      observe_state(t);
      observe_data_given_state(t);
    }
  }

  //----------------------------------------------------------------------
  void SSMB::set_state_model_behavior(StateModel::Behavior behavior) {
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->set_behavior(behavior);
    }
  }

  //----------------------------------------------------------------------
  void SSMB::allocate_full_kalman_filter(int number_of_time_points,
                                         int state_dimension) {
    if (number_of_time_points == 0) {
      full_kalman_storage_.clear();
      return;
    }
    if ((full_kalman_storage_.size() == (number_of_time_points + 1))
        && (full_kalman_storage_[0].a.size() == state_dimension)
        && (full_kalman_storage_.back().a.size() == state_dimension)) {
      return;
    }
    full_kalman_storage_.resize(number_of_time_points + 1,
                                ScalarKalmanStorage(state_dimension));
  }
  //----------------------------------------------------------------------
  void SSMB::check_light_kalman_storage(
      std::vector<LightKalmanStorage> &kalman_storage) {
    bool ok = true;
    if (kalman_storage.size() < time_dimension()) {
      kalman_storage.reserve(time_dimension());
      ok = false;
    }

    if (!kalman_storage.empty()) {
      if (kalman_storage[0].K.size() != state_dimension()) {
        kalman_storage.clear();
        ok = false;
      }
    }

    if (!ok) {
      for (int t = kalman_storage.size(); t < time_dimension(); ++t) {
        LightKalmanStorage s(state_dimension());
        kalman_storage.push_back(s);
      }
    }
  }

  //----------------------------------------------------------------------
  void SSMB::initialize_final_kalman_storage() const {
    ScalarKalmanStorage &ks(final_kalman_storage_);
    ks.a = initial_state_mean();
    ks.P = initial_state_variance();
    ks.F = observation_matrix(0).sandwich(ks.P) + observation_variance(0);
  }

  //----------------------------------------------------------------------
  void SSMB::register_data_observer(StateSpace::SufstatManagerBase *smb) {
    data_observers_.push_back(StateSpace::SufstatManager(smb));
  }

  namespace {
    // A functor that evaulates the log likelihood a
    // StateSpaceModelBase.  Suitable for passing to numerical
    // optimizers.
    class StateSpaceTargetFun {
     public:
      StateSpaceTargetFun(StateSpaceModelBase *model)
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
  double SSMB::mle(double epsilon) {
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

  bool SSMB::check_that_em_is_legal() const {
    if (!observation_model()->can_find_posterior_mode()) return false;
    for (int s = 0; s < nstate(); ++s) {
      if (!state_model(s)->can_find_posterior_mode()) {
        return false;
      }
    }
    return true;
  }

  //----------------------------------------------------------------------
  double SSMB::Estep(bool save_state_distributions) {
    return average_over_latent_data(true, save_state_distributions, nullptr);
  }

  //----------------------------------------------------------------------
  double SSMB::average_over_latent_data(
      bool update_sufficient_statistics,
      bool save_state_distributions,
      Vector *gradient) {
    if (update_sufficient_statistics) {
      clear_client_data();
    }
    std::vector<int> parameter_positions;
    if (gradient) {
      *gradient = vectorize_params(true) * 0.0;
    }
    // The call to full_kalman_filter() does two things.  First, it
    // sets log_likelihood_, which is the return value. Second, it
    // fills full_kalman_storage_ with filtered values.
    full_kalman_filter();

    Vector r(state_dimension(), 0.0);
    SpdMatrix N(state_dimension(), 0.0);
    for (int t = time_dimension() - 1; t >= 0; --t) {
      // From this point, up until
      // sparse_scalar_kalman_disturbance_smoother_update, r is r_t,
      // and N is N_t, using the notation from Durbin and Koopman.

      // Some syntactic sugar to make later formulas easier to read.
      // These are bad variable names, but they match the math in
      // Durbin and Koopman.
      const double H = observation_variance(t);
      const double F = full_kalman_storage_[t].F;
      const double v = full_kalman_storage_[t].v;
      const Vector &K(full_kalman_storage_[t].K);

      double u = v / F - K.dot(r);
      double D = (1.0 / F) + N.Mdist(K);

      const double observation_error_mean = H * u;
      const double observation_error_variance = H - H * D * H;
      if (save_state_distributions) {
        full_kalman_storage_[t].v = observation_error_mean;
        full_kalman_storage_[t].F = observation_error_variance;
      }
      if (update_sufficient_statistics) {
        update_observation_model_complete_data_sufficient_statistics(
            t, observation_error_mean, observation_error_variance);
      }
      if (gradient) {
        update_observation_model_gradient(
            observation_parameter_component(*gradient),
            t,
            observation_error_mean,
            observation_error_variance);
      }

      // Kalman smoother: convert r[t] to r[t-1] and N[t] to N[t-1].
      sparse_scalar_kalman_disturbance_smoother_update(
          r, N, (*state_transition_matrix(t)),
          K, observation_matrix(t), F, v);

      // The E step contribution for the observation at time t
      // involves the mean and the variance of the state error from
      // time t-1.
      //
      // The formula for the state error mean in Durbin and Koopman is
      // equation (4.41):   \hat \eta_t = Q_t R'_t r_t.
      //
      // state_error_mean is \hat eta[t-1]
      const Vector state_error_mean =
          (*state_error_variance(t-1)) * state_error_expander(t-1)->Tmult(r);

      // The formula for the state error variance in Durbin and
      // Koopman is equation (4.47):
      //
      // Var(\eta_t | Y) = Q - QR'NRQ  // all subscripted by _t
      //
      // state_error_posterior_variance is Var(\hat eta[t-1] | Y).
      SpdMatrix state_error_posterior_variance =
          state_error_expander(t-1)->sandwich_transpose(N);  // transpose??
      state_error_variance(t-1)->sandwich_inplace(state_error_posterior_variance);
      state_error_posterior_variance *= -1;
      state_error_variance(t-1)->add_to(state_error_posterior_variance);

      if (update_sufficient_statistics) {
        update_state_level_complete_data_sufficient_statistics(
            t-1,
            state_error_mean,
            state_error_posterior_variance);
      }

      if (gradient) {
        update_state_model_gradient(
            gradient,
            t-1,
            state_error_mean,
            state_error_posterior_variance);
      }

      if (save_state_distributions) {
        // Now r is r_{t-1} and N is N_{t-1}.  From Durbin and Koopman (4.32)
        // E(alpha[t] | Y) = a[t] + P * r[t-1]
        // V(alpha[t] | Y) = P[t] - P[t] * N[t-1] * P[t]
        SpdMatrix &P(full_kalman_storage_[t].P);
        full_kalman_storage_[t].a += (P * r);
        P -= sandwich(P, N);
      }

    }
    return log_likelihood_;
  }

  //----------------------------------------------------------------------
  void SSMB::Mstep(double epsilon) {
    observation_model()->find_posterior_mode(epsilon);
    for (int s = 0; s < nstate(); ++s) {
      state_model(s)->find_posterior_mode(epsilon);
    }
  }

  //----------------------------------------------------------------------
  Matrix SSMB::state_posterior_means() const {
    Matrix ans(state_dimension(), time_dimension());
    for (int t = 0; t < time_dimension(); ++t) {
      ans.col(t) = full_kalman_storage_[t].a;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Matrix SSMB::state_filtering_means() const {
    Matrix ans(state_dimension(), time_dimension());
    ans.col(0) = initial_state_mean();
    for (int t = 1; t < time_dimension(); ++t) {
      ans.col(t) = full_kalman_storage_[t-1].a;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  const SpdMatrix &SSMB::state_posterior_variance(int t) const {
    return full_kalman_storage_[t].P;
  }

  //----------------------------------------------------------------------
  Vector SSMB::observation_error_means() const {
    Vector ans(full_kalman_storage_.size());
    for (int i = 0; i < full_kalman_storage_.size(); ++i) {
      ans[i] = full_kalman_storage_[i].v;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  Vector SSMB::observation_error_variances() const {
    Vector ans(full_kalman_storage_.size());
    for (int i = 0; i < full_kalman_storage_.size(); ++i) {
      ans[i] = full_kalman_storage_[i].F;
    }
    return ans;
  }

  //----------------------------------------------------------------------
  // Send a signal to any object observing this model's data that
  // observation t has changed.
  void SSMB::signal_complete_data_change(int t) {
    for (int i = 0; i< data_observers_.size(); ++i) {
      data_observers_[i].update_complete_data_sufficient_statistics(t);
    }
  }

  //----------------------------------------------------------------------
  // Send a signal to any observers of this model's data that the
  // complete data sufficient statistics should be reset.
  void SSMB::signal_complete_data_reset() {
    for (int i = 0; i < data_observers_.size(); ++i) {
      data_observers_[i].clear_complete_data_sufficient_statistics();
    }
  }

}  // namespace BOOM

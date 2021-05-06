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

#include "Models/StateSpace/Filters/MultivariateKalmanFilterBase.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModelBase.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/Constants.hpp"

namespace BOOM {

  namespace Kalman {
    namespace {
      using Marginal = MultivariateMarginalDistributionBase;
    }

    void Marginal::set_forecast_precision_log_determinant(double logdet) {
      if (!std::isfinite(logdet)) {
        report_error("forecast precision is not finite.");
      }
      forecast_precision_log_determinant_ = logdet;
    }

    //--------------------------------------------------------------------------
    double Marginal::update(const Vector &observation,
                            const Selector &observed) {
      if (!model()) {
        report_error("ConditionalIidMarginalDistribution needs the model to be "
                     "set by set_model() before calling update().");
      }
      if (observed.nvars() == 0) {
        return fully_missing_update();
      }
      const SparseKalmanMatrix &transition(
          *model()->state_transition_matrix(time_index()));

      // The subset of observation coefficients corresponding to elements of
      // 'observation' which are actually observed.
      const SparseKalmanMatrix &observation_coefficient_subset(
          *model()->observation_coefficients(time_index(), observed));

      if (high_dimensional(observed)) {
        high_dimensional_update(observation, observed, transition,
                                observation_coefficient_subset);
      } else {
        low_dimensional_update(observation, observed, transition,
                               observation_coefficient_subset);
      }
      double log_likelihood = -.5 * observed.nvars() * Constants::log_root_2pi
          + .5 * forecast_precision_log_determinant()
          - .5 * prediction_error().dot(scaled_prediction_error());

      // Update the state mean from a[t]   = E(state_t    | Y[t-1]) to
      //                            a[t+1] = E(state[t+1] | Y[t]).
      set_state_mean(transition * state_mean()
                     + kalman_gain() * prediction_error());

      // Update the state variance from P[t] = Var(state_t | Y[t-1]) to P[t+1] =
      // Var(state[t+1] | Y[t]).
      //
      // The update formula is
      //
      // P[t+1] =   T[t] * P[t] * T[t]'
      //          - T[t] * P[t] * Z[t]' * K[t]'
      //          + R[t] * Q[t] * R[t]'
      //
      // Need to define TPZprime before modifying P (known here as
      // state_variance).
      Matrix TPZprime = (
          observation_coefficient_subset *
          (transition * state_variance()).transpose()).transpose();

      // Step 1:  Set P = T * P * T.transpose()
      transition.sandwich_inplace(mutable_state_variance());

      // Step 2:
      // Decrement P by T*P*Z.transpose()*K.transpose().  This step can be
      // skipped if y is missing, because K is zero.
      mutable_state_variance() -= TPZprime.multT(kalman_gain());

      // Step 3: P += RQR
      model()->state_variance_matrix(time_index())->add_to(
          mutable_state_variance());

      mutable_state_variance().fix_near_symmetry();
      return log_likelihood;
    }

    //----------------------------------------------------------------------
    bool Marginal::high_dimensional(const Selector &observed) const {
      return observed.nvars() > high_dimensional_threshold_factor()
          * model()->state_dimension();
    }

    //----------------------------------------------------------------------
    Vector Marginal::contemporaneous_state_mean() const {
      const Selector &observed(model()->observed_status(time_index()));
      if (!previous()) {
        return model()->initial_state_mean()
            + model()->initial_state_variance()
            * model()->observation_coefficients(0, observed)->Tmult(
                scaled_state_error());
      }
      return previous()->state_mean()
          + previous()->state_variance()
          * model()->observation_coefficients(time_index(), observed)->Tmult(
              scaled_state_error());
    }

    //----------------------------------------------------------------------
    SpdMatrix Marginal::contemporaneous_state_variance() const {
      SpdMatrix P = previous() ? model()->initial_state_variance()
          : previous()->state_variance();
      const Selector &observed(model()->observed_status(time_index()));
      const SparseKalmanMatrix *observation_coefficients(
          model()->observation_coefficients(time_index(), observed));
      return P - sandwich(
          P, observation_coefficients->sandwich_transpose(forecast_precision()));
    }

    //----------------------------------------------------------------------
    double Marginal::fully_missing_update() {
      // Compute the one-step prediction error and log likelihood contribution.
      const SparseKalmanMatrix  &transition(
          *model()->state_transition_matrix(time_index()));
      double log_likelihood = 0;
      set_prediction_error(Vector(0));

      // Update the state mean from a[t] = E(state_t | Y[t-1]) to a[t+1] =
      // E(state[t+1] | Y[t]).
      set_state_mean(transition * state_mean());

      // Update the state variance from P[t] = Var(state_t | Y[t-1]) to P[t+1] =
      // Var(state[t+1} | Y[t]).
      //
      // The update formula is
      //
      // P[t+1] = T[t] * P[t] * T[t]' + R[t] * Q[t] * R[t]'

      // Step 1:  Set P = T * P * T.transpose()
      transition.sandwich_inplace(mutable_state_variance());
      // Step 2: P += RQR
      model()->state_variance_matrix(time_index())->add_to(
          mutable_state_variance());
      mutable_state_variance().fix_near_symmetry();
      return log_likelihood;
    }

  }  // namespace Kalman

  //===========================================================================
  MultivariateKalmanFilterBase::MultivariateKalmanFilterBase(
      MultivariateStateSpaceModelBase *model)
      : model_(model) {}

  void MultivariateKalmanFilterBase::update() {
    if (!model_) {
      report_error("Model must be set before calling update().");
    }
    clear_loglikelihood();
    for (int t = 0; t < model_->time_dimension(); ++t) {
      update_single_observation(
          model_->adjusted_observation(t),
          model_->observed_status(t),
          t);
      if (!std::isfinite(log_likelihood())) {
        set_status(NOT_CURRENT);
        return;
      }
    }
    set_status(CURRENT);
  }

  void MultivariateKalmanFilterBase::update_single_observation(
      const Vector &y,
      const Selector &observed,
      int t) {
    if (!model_) {
      report_error("Model must be set before calling update().");
    }
    ensure_size(t);
    if (t == 0) {
      node(t).set_state_mean(model_->initial_state_mean());
      node(t).set_state_variance(model_->initial_state_variance());
    } else {
      node(t).set_state_mean(node(t - 1).state_mean());
      node(t).set_state_variance(node(t - 1).state_variance());
    }
    increment_log_likelihood(node(t).update(y, observed));
  }

  // Disturbance smoother replaces Durbin and Koopman's K[t] with r[t].  The
  // disturbance smoother is equation (5) in Durbin and Koopman (2002,
  // Biometrika).
  //
  // Returns:
  //   Durbin and Koopman's r0.  Saves r[t] in node(t).scaled_state_error().
  void MultivariateKalmanFilterBase::fast_disturbance_smooth() {
    if (!model_) {
      report_error("Model must be set before calling fast_disturbance_smooth().");
    }

    int n = model_->time_dimension();
    Vector r(model_->state_dimension(), 0.0);
    for (int t = n - 1; t >= 0; --t) {
      // Currently r is r[t].  This step of the loop turns it into r[t-1].
      //
      // The disturbance smoother is defined by the following formula:
      // r[t-1] = Z' * Finv * v   +   (T' - Z' * K') * r[t]
      //        = T' * r[t]       -   Z' * (K' * r[t] - Finv * v)
      //
      // Note that Durbin and Koopman (2002) is missing the transpose on Z in
      // their equation (5).  The transpose is required to get the dimensions to
      // match.
      //
      // If we stored (Z' * K') that would only be SxS.  Maybe put some smarts
      // in depending on whether m or S is larger.
      //
      // K = TPZ'Finv
      // Z' K' = Z' Finv Z P T'
      //
      // Dimensions:
      //   T:    S x S
      //   K:    S x m
      //   Z:    m x S
      //   Finv: m x m
      //   v:    m x 1
      //   r:    S x 1
      //
      node(t).set_scaled_state_error(r);
      const Selector &observed(model_->observed_status(t));
      r = model_->state_transition_matrix(t)->Tmult(r)
          - model_->observation_coefficients(t, observed)->Tmult(
              node(t).kalman_gain().Tmult(r) - node(t).scaled_prediction_error());
    }
    set_initial_scaled_state_error(r);
  }

  Vector MultivariateKalmanFilterBase::prediction_error(int t, bool standardize) const {
    if (standardize) {
      return (*this)[t].scaled_prediction_error();
    } else {
      return (*this)[t].prediction_error();
    }
  }

}  // namespace BOOM

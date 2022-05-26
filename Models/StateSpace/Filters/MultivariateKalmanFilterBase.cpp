/*
  Copyright (C) 2005-2022 Steven L. Scott

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
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "cpputil/report_error.hpp"
#include "cpputil/Constants.hpp"

namespace BOOM {

  namespace Kalman {
    void check_variance(const SpdMatrix &v) {
      for (auto x : v.diag()) {
        if (x < -.10) {
          report_error("Can't have a negative variance.");
        }
      }
    }

    namespace {
      using Marginal = MultivariateMarginalDistributionBase;
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
      Ptr<SparseKalmanMatrix> observation_coefficient_pointer(
          model()->observation_coefficients(time_index(), observed));
      const SparseKalmanMatrix &observation_coefficient_subset(
          *observation_coefficient_pointer);

      Vector observed_data = observed.select(observation);
      set_prediction_error(
          observed_data - observation_coefficient_subset * state_mean());
      update_sparse_forecast_precision(observed);

      const SparseBinomialInverse &Finv(*sparse_forecast_precision());

      double log_likelihood = -.5 * observed.nvars() * Constants::log_root_2pi
          + .5 * forecast_precision_log_determinant()
          - .5 * prediction_error().dot(scaled_prediction_error());

      Ptr<SparseMatrixProduct> gain = sparse_kalman_gain(observed);
      const SparseMatrixProduct &kalman_gain(*gain);

      // Update the state mean from a[t]   = E(state_t    | Y[t-1]) to
      //                            a[t+1] = E(state[t+1] | Y[t]).
      set_state_mean(transition * state_mean()
                     + kalman_gain * prediction_error());

      // Update the state variance.  This implementation is the result of some
      // debugging.  If profiling shows too much time is spent in this function,
      // this section is a good place to look for optimizations.  We probably
      // don't need to allocate quite so many SpdMatrix objects.
      //
      // Pt|t = Pt - Pt * Z' Finv Z Pt
      SpdMatrix new_state_variance = state_variance();

      SpdMatrix increment1 = state_variance() * observation_coefficient_subset.Tmult(
          Finv * (observation_coefficient_subset * state_variance()));

      transition.sandwich_inplace(new_state_variance);
      model()->state_variance_matrix(time_index())->add_to(new_state_variance);

      SpdMatrix contemp_variance(state_variance() - increment1);
      Kalman::check_variance(contemp_variance);

      SpdMatrix increment2(model()->state_variance_matrix(time_index())->dense());

      new_state_variance = contemp_variance;
      transition.sandwich_inplace(new_state_variance);
      new_state_variance += increment2;
#ifndef NDEBUG
      // Only check the variance in debug mode.
      Kalman::check_variance(new_state_variance);
#endif

      set_state_variance(new_state_variance);
      return log_likelihood;
    }

    //----------------------------------------------------------------------
    Vector Marginal::scaled_prediction_error() const {
      return (*sparse_forecast_precision()) * prediction_error();
    }

    //----------------------------------------------------------------------
    Ptr<SparseMatrixProduct> Marginal::sparse_kalman_gain(
        const Selector &observed) const {
      NEW(SparseMatrixProduct, ans)();
      int t = time_index();
      ans->add_term(model()->state_transition_matrix(t));
      // Going forward, previous()->state_variance() and state_variance() will be
      // the same.  Going backward they may be different.
      const Marginal *prev = previous();
      NEW(DenseSpd, P)(prev ? previous()->state_variance()
                       : model()->initial_state_variance());
      ans->add_term(P);
      ans->add_term(model()->observation_coefficients(t, observed), true);
      ans->add_term(sparse_forecast_precision());
      return ans;
    }

    //----------------------------------------------------------------------
    Vector Marginal::contemporaneous_state_mean() const {
      const Selector &observed(model()->observed_status(time_index()));
      const Marginal *prev = previous();
      if (!prev) {
        return model()->initial_state_mean()
            + model()->initial_state_variance()
            * model()->observation_coefficients(0, observed)->Tmult(
                scaled_state_error());
      } else{
        return prev->state_mean()
            + prev->state_variance() * model()->observation_coefficients(
                time_index(), observed)->Tmult(scaled_state_error());
      }
    }

    //----------------------------------------------------------------------
    SpdMatrix Marginal::contemporaneous_state_variance() const {
      const Marginal *prev = previous();
      SpdMatrix P = prev ? prev->state_variance() :
          model()->initial_state_variance();
      const Selector &observed(model()->observed_status(time_index()));
      Ptr<SparseKalmanMatrix> observation_coefficients(
          model()->observation_coefficients(time_index(), observed));
      NEW(SparseMatrixProduct, ZFZ)();
      ZFZ->add_term(observation_coefficients, true);
      ZFZ->add_term(sparse_forecast_precision(), false);
      ZFZ->add_term(observation_coefficients, false);
      return P - sandwich(P, SpdMatrix(ZFZ->dense()));
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
  void MultivariateKalmanFilterBase::update() {
    if (!model()) {
      report_error("Model must be set before calling update().");
    }
    clear_loglikelihood();
    // std::cout << "model->observation_coefficients() = \n";
    // model()->observation_coefficients(0, model()->observed_status(0))->print(std::cout);

    // TODO: Verify that the isolate_shared_state line doesn't break anything
    // when the model has series-specific state.
    model_->isolate_shared_state();
    for (int time = 0; time < model_->time_dimension(); ++time) {
      update_single_observation(
          model_->adjusted_observation(time),
          model_->observed_status(time),
          time);
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
      Kalman::MultivariateMarginalDistributionBase &marg(node(t));
      marg.set_scaled_state_error(r);

      // All implicit subsctipts are [t].
      //  r[t-1] = Z' * Finv * v - L' r
      //  where
      //  L[t] = T[t] - K[t] Z[t]
      // so
      // r[t-1] = Z' * Finv * v - T'r + Z'K'r
      //
      // u = Finv * v - K'r
      // r = T'r + Z'u
      const Selector &observed(model_->observed_status(t));
      Ptr<SparseKalmanMatrix> observation_coefficients(
          model_->observation_coefficients(t, observed));
      Ptr<SparseKalmanMatrix> transition(
          model_->state_transition_matrix(t));
      const SparseBinomialInverse &forecast_precision(
          *marg.sparse_forecast_precision());
      Vector u = forecast_precision * marg.prediction_error()
          - marg.sparse_kalman_gain(observed)->Tmult(r);
      r = transition->Tmult(r) + observation_coefficients->Tmult(u);
    }
    set_initial_scaled_state_error(r);
  }

  Vector MultivariateKalmanFilterBase::prediction_error(
      int t, bool standardize) const {
    const auto &marginal((*this)[t]);
    if (standardize) {
      return *marginal.sparse_forecast_precision()
          * marginal.prediction_error();
    } else {
      return marginal.prediction_error();
    }
  }

}  // namespace BOOM

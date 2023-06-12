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
#include "LinAlg/Eigen.hpp"

namespace BOOM {

  namespace Kalman {
    void check_variance(const SpdMatrix &v) {
#ifndef NDEBUG
      for (auto x : v.diag()) {
        if (x < .00) {
          report_error("Can't have a negative variance.");
        }
      }
# endif
    }

    namespace {
      using Marginal = MultivariateMarginalDistributionBase;

      // Convert a Matrix to an SpdMatrix.  If the matrix is non-symmetric then
      // force symmetry and optionally issue a warning.
      //
      // Args:
      //   input_matrix:  The Matrix to convert to an SpdMatrix.
      //   time:  The time point at which the conversion was attempted.
      //   warn: Should a warning be printed if mitigating steps were needed to
      //     make the conversion?
      SpdMatrix robust_spd(const Matrix &input_matrix, int time, bool show_warnings) {
        if (input_matrix.is_sym()) {
          return SpdMatrix(input_matrix);
        } else {
          if (show_warnings) {
            std::ostringstream msg;
            double distance;
            uint imax, jmax;
            std::tie(distance, imax, jmax) =
                input_matrix.distance_from_symmetry();
            msg << "Coercing a non-symmetric matrix to symmetry at time "
                << time << ".\n"
                << "Distance from symmetry = " << distance << " with maximum "
                "relative distance at (" << imax
                << ", " << jmax << ").\n";
            if (distance > .01) {
              if (input_matrix.nrow() < 10) {
                msg << "\n"
                    << "original matrix: \n"
                    << input_matrix
                    << "\n"
                    << "symmetric matrix: \n"
                    << .5 * (input_matrix + input_matrix.transpose())
                    ;
              } else {
                Matrix m_view = ConstSubMatrix(
                    input_matrix, 0, 9, 0, 9).to_matrix();
                msg << "\n"
                    << "First 10 rows/cols of original matrix:\n"
                    << m_view
                    << "\n"
                    << "symmetric matrix:\n"
                    << .5 * (m_view + m_view.transpose());
              }
            }
            report_warning(msg.str());
          }
          return SpdMatrix(.5 * (input_matrix + input_matrix.transpose()));
        }
      }
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

      Ptr<SparseKalmanMatrix> transition_ptr(
          model()->state_transition_matrix(time_index()));
      const SparseKalmanMatrix &transition(*transition_ptr);

      // The subset of observation coefficients corresponding to elements of
      // 'observation' which are actually observed.
      Ptr<SparseKalmanMatrix> observation_coefficient_pointer(
          model()->observation_coefficients(time_index(), observed));
      const SparseKalmanMatrix &observation_coefficient_subset(
          *observation_coefficient_pointer);

      Vector observed_data = observed.select_if_needed(observation);
      set_prediction_error(
          observed_data - observation_coefficient_subset * state_mean());
      update_sparse_forecast_precision(observed);

      Ptr<SparseKalmanMatrix> Finv_ptr(sparse_forecast_precision());
      const SparseKalmanMatrix &Finv(*Finv_ptr);

      double log_likelihood = -.5 * observed.nvars() * Constants::log_root_2pi
          + .5 * forecast_precision_log_determinant()
          - .5 * prediction_error().dot(Finv * prediction_error());
      if (std::isnan(log_likelihood)) {
        // This line is important when the model is being fit by optimization.
        // If a variance becomes negative (or negative definite) during part of
        // the optimization algorithm, then the function value should be set to
        // negative infinity so that the optimization algorithm will know it
        // took a bad step.  NaN cannot be less-than compared, so NaN values
        // will just confuse an optimizer.
        log_likelihood = negative_infinity();
      }

      Ptr<SparseMatrixProduct> gain = sparse_kalman_gain(observed, Finv_ptr);
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

      // The temporary value 'tmp' is needed because the long string of
      // multiplications can produce temporaries that are not symmetric.  The
      // system is free to try to optimize this multiplication using different
      // associations.  If we try to define an SpdMatrix to receive the outcome,
      // non-symmetric temporaries can blow up the SpdMatrix constructor.
      Matrix increment1 = state_variance() * observation_coefficient_subset.Tmult(
          Finv * (observation_coefficient_subset * state_variance()));

      SpdMatrix contemp_variance(
          robust_spd(state_variance() - increment1,
                     time_index(),
                     model()->show_warnings()));
      if (!contemp_variance.is_pos_def()) {
        SymmetricEigen contemp_eigen(contemp_variance, true);
        SpdMatrix updated = contemp_eigen.closest_positive_definite();
        if (model()->show_warnings()) {
          std::ostringstream warn;
          warn << "Modifying variance at time " << time_index()
               << " to enforce positive definiteness.\n";
          int imax, jmax;
          double distance = relative_distance(
              contemp_variance, updated, imax, jmax);
          if (distance > .001) {
            warn << "Original matrix:\n" << contemp_variance
                 << "Updated matrix: \n" << updated;
          }
          warn << "Distance = " << distance
               << ".  Maximum relative deviation in position ("
               << imax << ", " << jmax << ").\n";
          report_warning(warn.str());
        }
        contemp_variance = updated;
      }

      SpdMatrix increment2(
          robust_spd(model()->state_variance_matrix(time_index())->dense(),
                     time_index(),
                     model()->show_warnings()));
      SpdMatrix new_state_variance(
          robust_spd(contemp_variance,
                     time_index(),
                     model()->show_warnings()));

      transition.sandwich_inplace(new_state_variance);

      new_state_variance += increment2;
      set_state_variance(new_state_variance);

      return log_likelihood;
    }

    //----------------------------------------------------------------------
    Ptr<SparseMatrixProduct> Marginal::sparse_kalman_gain(
        const Selector &observed,
        const Ptr<SparseKalmanMatrix> &forecast_precision) const {
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
      ans->add_term(forecast_precision);
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
    SpdMatrix Marginal::contemporaneous_state_variance(
        const Ptr<SparseKalmanMatrix> &forecast_precision) const {
      const Marginal *prev = previous();
      SpdMatrix P = prev ? prev->state_variance() :
          model()->initial_state_variance();
      const Selector &observed(model()->observed_status(time_index()));
      Ptr<SparseKalmanMatrix> observation_coefficients(
          model()->observation_coefficients(time_index(), observed));
      NEW(SparseMatrixProduct, ZFZ)();
      ZFZ->add_term(observation_coefficients, true);
      ZFZ->add_term(forecast_precision, false);
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
    // TODO: Verify that the isolate_shared_state line doesn't break anything
    // when the model has series-specific state.
    model()->isolate_shared_state();
    ensure_size(model()->time_dimension());
    for (int time = 0; time < model()->time_dimension(); ++time) {
      update_single_observation(
          model()->adjusted_observation(time),
          model()->observed_status(time),
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
    if (!model()) {
      report_error("Model must be set before calling update().");
    }
    ensure_size(t);
    if (t == 0) {
      node(t).set_state_mean(model()->initial_state_mean());
      node(t).set_state_variance(model()->initial_state_variance());
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
    if (!model()) {
      report_error("Model must be set before calling fast_disturbance_smooth().");
    }

    int n = model()->time_dimension();
    Vector r(model()->state_dimension(), 0.0);
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
      const Selector &observed(model()->observed_status(t));
      Ptr<SparseKalmanMatrix> transition(
          model()->state_transition_matrix(t));
      if (observed.nvars() > 0) {
        Ptr<SparseKalmanMatrix> observation_coefficients(
            model()->observation_coefficients(t, observed));
        Ptr<SparseKalmanMatrix> forecast_precision_ptr(
            marg.sparse_forecast_precision());
        const SparseKalmanMatrix &forecast_precision(
            *forecast_precision_ptr);
        Vector u = forecast_precision * marg.prediction_error()
            - marg.sparse_kalman_gain(observed, forecast_precision_ptr)->Tmult(r);
        r = transition->Tmult(r) + observation_coefficients->Tmult(u);
      } else {
        r = transition->Tmult(r);
      }
    }
    set_initial_scaled_state_error(r);
  }

  //===========================================================================
  void MultivariateKalmanFilterBase::smooth() {
    // All implicit subsctipts are [t].
    //  r[t-1] = Z' * Finv * v - L' r
    //  where
    //  L[t] = T[t] - K[t] Z[t]
    // so
    // r[t-1] = Z' * Finv * v - T'r + Z'K'r
    //
    // N[t-1] = Z' Finv Z + L' N L
    //
    // smoothed_state_mean[t] = a[t] + P[t] * r[t-1]
    // smoothed_state_variance[t] = P[t] - P[t] N[t-1] P[t]
    //
    // The algorithm is initialized by r[T] = 0, and N[T] = 0.

    if (!model()) {
      report_error("Model must be set before calling fast_disturbance_smooth().");
    }

    int n = model()->time_dimension();
    int state_dimension = model()->state_dimension();
    Vector r(state_dimension, 0.0);
    Matrix N(state_dimension, state_dimension, 0.0);
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

      // All implicit subsctipts are [t].
      //  r[t-1] = Z' * Finv * v - L' r
      //  where
      //  L[t] = T[t] - K[t] Z[t]
      // so
      // r[t-1] = Z' * Finv * v - T'r + Z'K'r
      //
      // u = Finv * v - K'r
      // r = T'r + Z'u
      Ptr<SparseKalmanMatrix> transition(
          model()->state_transition_matrix(t));
      const Selector &observed(model()->observed_status(t));
      if (observed.nvars() > 0) {
        Ptr<SparseKalmanMatrix> observation_coefficients(
            model()->observation_coefficients(t, observed));
        Ptr<SparseKalmanMatrix> forecast_precision_ptr(
            marg.sparse_forecast_precision());
        const SparseKalmanMatrix &forecast_precision(
            *forecast_precision_ptr);
        Ptr<SparseKalmanMatrix> kalman_gain(marg.sparse_kalman_gain(
            observed, forecast_precision_ptr));
        NEW(SparseMatrixProduct, KZ)();
        KZ->add_term(kalman_gain);
        KZ->add_term(observation_coefficients);
        NEW(SparseMatrixSum, L)();
        L->add_term(transition);
        L->add_term(KZ, -1);
        Vector u = forecast_precision * marg.prediction_error()
            - kalman_gain->Tmult(r);

        // Turn r[t] into r[t-1]
        r = transition->Tmult(r) + observation_coefficients->Tmult(u);
        // Turn N[t] into N[t-1]
        Matrix tmp = observation_coefficients->Tmult(
            forecast_precision * observation_coefficients->dense())
            + L->sandwich_transpose(N);
        N = tmp;
      } else {
        r = transition->Tmult(r);
        N = transition->sandwich_transpose(N);
      }

      Vector filtered_state_mean;
      SpdMatrix filtered_state_variance;
      if (t > 0) {
        Kalman::MultivariateMarginalDistributionBase &prev(node(t-1));
        filtered_state_mean = prev.state_mean();
        filtered_state_variance = prev.state_variance();
      } else {
        filtered_state_mean = model()->initial_state_mean();
        filtered_state_variance = model()->initial_state_variance();
      }

      Vector smoothed_state_mean =
          filtered_state_mean + filtered_state_variance * r;

      if (t == 0 && !N.is_sym()) {
        // TODO:
        //   Sometimes N is not symmetric at t==0.  The right way to fix this is
        //   to have the initial distribution be a Marg object that can get
        //   updated.  The following line is a stop-gap for now.
        N = .5 * (N + N.transpose());
      }
      SpdMatrix SpdN(Kalman::robust_spd(N, t, model()->show_warnings()));
      if (!SpdN.is_pos_def()) {
        SymmetricEigen eigenN(SpdN);
        SpdN = eigenN.closest_positive_definite();
      }

      SpdMatrix smoothed_state_variance = Kalman::robust_spd(
          filtered_state_variance - sandwich(filtered_state_variance, SpdN),
          t,
          model()->show_warnings());

      if (!smoothed_state_variance.is_pos_def()) {
        SymmetricEigen variance_eigen(smoothed_state_variance);
        smoothed_state_variance = variance_eigen.closest_positive_definite();
      }

      marg.set_state_mean(smoothed_state_mean);
      marg.set_state_variance(smoothed_state_variance);
      N = SpdN;
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

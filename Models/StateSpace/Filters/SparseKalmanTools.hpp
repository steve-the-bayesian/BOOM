/*
  Copyright (C) 2008-2011 Steven L. Scott

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

#ifndef BOOM_SPARSE_KALMAN_TOOLS_HPP
#define BOOM_SPARSE_KALMAN_TOOLS_HPP

#include <LinAlg/SpdMatrix.hpp>

#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>

namespace BOOM{
  // Returns the likelihood contribution of y given previous y's.
  // Uses notation from Durbin and Koopman (2001):
  //
  //     y[t] = Z[t].dot(alpha[t]) + epsilon[t] ~ N(0, H[t])
  // alpha[t] = T[t] * alpha[t-1] + R[t]eta[t] ~ N(0, RQR.transpose())
  //
  // In the code below...
  //   a[t] = E(alpha[t]| Y^{t-1}) is the one-step ahead expected
  //     value of the state at time t given data to time t-1.
  //   P[t] = V(alpha[t]| Y^{t-1}) is the conditional variance of the
  //     state at time t given data to time t-1.
  //   v[t] is the one step forecast error of y[t] | Y^{t-1}, y[t] -
  //     Z[t].dot(a[t]).  Note that v[t] is the argument of the
  //     density used to compute the incremental log likelihood
  //     contribution from observation t.
  //        p(y[t] | Y[t-1]) = N(y[t] |a[t], F[t])
  //                         = N(v[t] | 0, F[t])
  //   K[t] is the "Kalman Gain:" The update equation for the forecast
  //     mean is a[t+1] = T[t]*a[t] + K[t]*v[t], which one can
  //     recognize as a regression, with T[t]*a[t] being the intercept
  //     term, with K[t] playing the role of "slope" and v[t] the new
  //     information supplied by y[t].
  //   F[t] = Var(y[t] | Y^{t-1}) is the conditional variance of y[t]
  //     given data to time t-1.  In addition to simply being needed
  //     for computational reasons, this is the variance needed to
  //     evaluate the likelihood.
  //
  // Args:
  //   y: The observed value of y[t].  If y is missing an arbitrary
  //     value can be supplied, as it will not be used.
  //   a: On input this is a[t], as noted above.  On output 'a' gets
  //     promoted to a[t+1].
  //   P:  On input this is P[t].  On output it is promoted to P[t+1].
  //   kalman_gain:  Input is not read.  Output is K[t].
  //   forecast_error_variance:  Input is not read.  Output is F[t].
  //   forecast_error:  Input is not read.  Output is v[t].
  //   missing: If 'true' then the 'y' argument is taken to be
  //     missing.  Otherwise 'y' is taken as observed.
  //   Z: The vector from the observation linking the state at time t
  //     to the observation at time t.  y[t] = Z[t].dot(state[t]).
  //   observation_variance: The variance of the observation equation
  //     at time t.
  //   T: The matrix from the transition equation at time t, linking
  //     the state at time t to the state at time t+1.  state[t+1] =
  //     T[t] * state[t] + error[t].
  //   RQR: The (possibly rank deficient) error variance of the random
  //     innovations in the transition equation at time t (linking the
  //     state at time t to the state at time t+1).  state[t+1] =
  //     T[t] * state[t] + error[t], where error[t] ~ N(0, RQR[t]).
  //
  // Returns:
  //   This observation's contribution to log likelihood.
  double sparse_scalar_kalman_update(
      double y,                      // y[t]
      Vector &a,                     // a[t] -> a[t+1]
      SpdMatrix &P,                  // P[t] -> P[t+1]
      Vector &kalman_gain,           // output as K[t]
      double &forecast_error_variance, // output as F[t]
      double &forecast_error,          // output as v[t]
      bool missing,                   // was y observed?
      const SparseVector &Z,  // input
      double observation_variance,
      const SparseKalmanMatrix &T,
      const SparseKalmanMatrix &RQR);   // state transition error variance

  // Updates a[t] and P[t] to condition on all Y, and sets up r and N
  // for use in the next recursion.
  void sparse_scalar_kalman_smoother_update(
      Vector &a,                   // a[t] -> E(alpha[t] | Y)
      SpdMatrix &P,                // P[t] -> V(alpha[t] | Y)
      const Vector &K,             // K[t] As produced by Kalman filter
      double forecast_variance,    // F[t] "
      double forecast_error,       // v[t] "
      const SparseVector &Z,       // Z[t] "
      const SparseKalmanMatrix &T, // T[t] "
      Vector &r,                   // backward Kalman variable, local
      Matrix &N);                  // backward Kalman variance, local


  struct DisturbanceSmoothingStorage {
    // Scaled, potentially rank-deficient version of the state
    // innovation eta.  Note that 'r' has dimension equal to 'state'
    // while 'eta' has potentially lower dimension.  Related by the
    // equation eta-hat[t] = Q[t] * R[t].transpose() * r[t], where Q
    // and R are defined by the transition equation:
    //   state[t+1] = T[t] * state[t] + R[t] * eta[t]
    //   with (eta[t] ~ N(0,Q[t]))
    Vector r;

    // Variance of r.
    SpdMatrix N;
  };

  // Args:
  //   scaled_residual_r: On input this is the expected value of the
  //     scaled state-residual from time t, given all data.  On
  //     output it is the expected scaled state residual from time t-1.
  //   scaled_residual_variance_N: On input this is the conditional
  //     variance of the scaled state residual at time t, given all
  //     data.  On output it is the variance of r at time t-1.
  //   transition_matrix_T: The 'T' matrix from the transition
  //     equation at time t.
  //   kalman_gain_K: The kalman gain from time t, originally produced
  //     by the Kalman filter.
  //   observation_matrix_Z: The matrix from the observation equation
  //     at time t.
  //   forecast_variance: The forecast variance, or conditional
  //     variance of y[t] given Y[t-1], produced by the Kalman filter.
  //   forecast_error: The one step forecast error
  //     y[t] - E(y[t] | Y[t-1]), produced by the Kalman filter.
  //
  // Side effects:
  //   This function is called to produce updates of scaled_residual_r
  //   and scaled_residual_variance_N.
  //
  // Initial values.  The backward smoothing recursions are started
  // with r[T+1] = 0 and N[T+1] = 0.
  void sparse_scalar_kalman_disturbance_smoother_update(
      Vector &scaled_residual_r,
      SpdMatrix &scaled_residual_variance_N,
      const SparseKalmanMatrix &transition_matrix_T,
      const Vector &kalman_gain_K,
      const SparseVector &observation_matrix_Z,
      double forecast_variance,
      double forecast_error);

}  // namespace BOOM
#endif// BOOM_SPARSE_KALMAN_TOOLS_HPP

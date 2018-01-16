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

#include <Models/StateSpace/Filters/KalmanTools.hpp>
#include <Models/StateSpace/Filters/SparseKalmanTools.hpp>
#include <Models/StateSpace/Filters/SparseVector.hpp>
#include <Models/StateSpace/Filters/SparseMatrix.hpp>
#include <distributions.hpp>
#include <cpputil/report_error.hpp>

namespace BOOM{
  double sparse_scalar_kalman_update(
      double y,                         // New observation at time t
      Vector &a,                        // Input a[t].  Output a[t+1]
      SpdMatrix &P,                     // Input P[t].  Ouput P[t+1]
      Vector &K,                        // Output K[t] Kalman gain
      double &F,                        // Output F[t] Forecast variance
      double &v,                        // Output v[t] Forecast error
      bool missing,                     // true if y[t] is missing
      const SparseVector & Z,           // Model matrix for obs. equation
      double H,                         // Var(Y | state)
      const SparseKalmanMatrix & T,     // State transition matrix
      const SparseKalmanMatrix & RQR) { // State variance matrix

    Vector PZ = P*Z;
    F = Z.dot(PZ) + H;
    if (F <= 0) {
      std::ostringstream err;
      err << "Found a zero forecast variance:" << endl
          << "missing = " << missing << endl
          << "a = " << a << endl
          << "P = " << endl << P << endl
          << "y = " << y << endl
          << "H = " << H << endl
          << "ZPZ = " << Z.dot(PZ) << endl
          << "Z = " << Z.dense() << endl;
      report_error(err.str());
    }
    Vector TPZ = T * PZ;

    double loglike=0;
    if (!missing) {
      K = TPZ/F;
      double mu = Z.dot(a);
      v = y-mu;
      loglike = dnorm(y, mu, sqrt(F), true);
    }else{
      K = a.zero();
      v = 0;
    }

    a = T * a;                         // Sparse multiplication
    if (!missing) a.axpy(K, v);         // a += K * v
    T.sandwich_inplace(P);             // P = T P T.transpose()
    if (!missing) {                      // K is zero if missing, so skip this
      P.Matrix::add_outer(TPZ, K, -1); // P-= T*P*Z*K.transpose();
    }
    RQR.add_to(P);                     // P += RQR

    return loglike;
  }

  // As part of the Kalman smoothing (backward) recursion, update the
  // vector r[t] and the matrix N[t] to time t-1.
  //
  // Args:
  //   The arguments below correspond to elements in Durbin and
  //     Koopman, second edition, page 95, Section 4.5.3, in the
  //     equation following (4.69).
  //   scaled_residual_r: On input this is r[t].  On output it is
  //     r[t-1].
  //   scaled_residual_variance_N: On input this is N[t], on output it
  //     is N[t-1].
  //   transition_matrix_T:  T[t]
  //   kalman_gain_K: K[t]
  //   observation_matrix_Z:  Z[t]'
  //   forecast_variance:  F[t]
  //   forecast_error: v[t]
  void sparse_scalar_kalman_disturbance_smoother_update(
      Vector &scaled_residual_r,
      SpdMatrix &scaled_residual_variance_N,
      const SparseKalmanMatrix &transition_matrix_T,
      const Vector &kalman_gain_K,
      const SparseVector &observation_matrix_Z,
      double forecast_variance,
      double forecast_error) {

    // The recursion is given by the following system of equations:
    // u[t]    = (v[t] / F[t]) - K[t] * r[t]
    // r[t-1]  = (Z[t] * u[t]) + (T[t]' * r[t])
    // D[t]    = (1.0 / F[t]) + K[t]' * N[t] * K[t]
    // N[t-1]  = ZDZ' + T'NT - ZK'NT - T'NKZ'  // all [t]

    // u[t] = F[t]^{-1} * v[t] - K[t].dot(r[t])
    double u = forecast_error / forecast_variance
        - kalman_gain_K.dot(scaled_residual_r);
    // r[t-1] = T'r + Z*u
    Vector previous_r = transition_matrix_T.Tmult(scaled_residual_r);
    observation_matrix_Z.add_this_to(previous_r, u);
    scaled_residual_r = previous_r;

    // D = (1/F) + K'NK
    double D = (1.0 / forecast_variance)
        + scaled_residual_variance_N.Mdist(kalman_gain_K);

    SpdMatrix previousN = scaled_residual_variance_N;
    transition_matrix_T.sandwich_inplace_transpose(previousN);
    // N[t-1] = T'NT
    observation_matrix_Z.add_outer_product(previousN, D);
    // N[t-1] = ZDZ' + T'NT

    Vector TprimeNK = transition_matrix_T.Tmult(
        scaled_residual_variance_N * kalman_gain_K);
    Matrix TprimeNKZ = observation_matrix_Z.outer_product_transpose(TprimeNK);
    // Next, previousN = previousN - TprimeNKZ -
    // TprimeNKZ.transpose().  Subtracting the transpose column by
    // column should maximize the use of blas routines.
    previousN -= TprimeNKZ;
    for (int i = 0; i < ncol(previousN); ++i) {
      previousN.col(i) -= TprimeNKZ.row(i);
    }
    scaled_residual_variance_N = previousN;
  }

}

// Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef BOOM_KALMAN_TOOLS_HPP
#define BOOM_KALMAN_TOOLS_HPP

#include "LinAlg/SpdMatrix.hpp"

namespace BOOM {
  // Returns the likelihood contribution of y given previous y's.
  // Uses notation from Durbin and Koopman (2001).
  // y is y[t]
  // a starts as a[t] and ends as a[t+1]  a[t] = E(alpha[t]| Y^{t-1})
  // P starts as P[t] and ends as P[t+1]  P[t] = V(alpha[t]| Y^{t-1})
  // K is output as K[t]
  // Finv is output as Finv[t]
  // v is output as v[t]
  double scalar_kalman_update(double y,      // y[t]
                              Vector &a,     // a[t] -> a[t+1]
                              SpdMatrix &P,  // P[t] -> P[t+1]
                              Vector &K,     // output as K[t]
                              double &F,     // output as F[t]
                              double &v,     // output as v[t]
                              bool missing, const Vector &Z, double H,
                              const Matrix &T, Matrix &L, const SpdMatrix &RQR);

  // The Kalman filter as implemented above computes the predictive
  // distribution of the state at time t+1 given data up to time t.
  // This function takes the outputs of scalar_kalman_update and
  // converts the mean and variance so that they refer to the state at
  // time t (rather than t+1) given data up to time t.
  //
  // Args:
  //   a: On input this is the mean of the state at time t+1 given
  //     data to time t.  On output it is the mean of the state at
  //     time t given data to time t.
  //   P: On input this is the variance of the state at time t+1 given
  //     data to time t.  On output it is the variance of the state at
  //     time t given data at time t.
  //   F_forecast_variance: This is the one_step ahead forecast
  //     variance at time t (the output 'F' of scalar_kalman_update
  //     upon observing y[t]).
  //   v_one_step_prediction_error: The one step prediction error at
  //     time t (the output 'v' of scalar_kalman_update upon observing
  //     y[t]).
  //   Z_state_reducer:  The value of Z passed to scalar_kalman_update.
  void make_contemporaneous(Vector &a, SpdMatrix &P, double F_forecast_variance,
                            double v_one_step_prediction_error,
                            const Vector &Z_state_reducer);

  // Updates a[t] and P[t] to condition on all Y, and sets up r and N
  // for use in the next recursion.
  void scalar_kalman_smoother_update(Vector &a, SpdMatrix &P, const Vector &K,
                                     double F, double v, const Vector &Z,
                                     const Matrix &T, Vector &r, Matrix &N,
                                     Matrix &L);

}  // namespace BOOM
#endif  // BOOM_KALMAN_TOOLS_HPP

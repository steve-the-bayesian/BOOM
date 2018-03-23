// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008-2017 Steven L. Scott

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

#ifndef BOOM_KALMAN_STORAGE_HPP
#define BOOM_KALMAN_STORAGE_HPP

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  // Store the conditional state mean and conditional state variance.
  struct KalmanStateStorage {
    KalmanStateStorage() {}
    explicit KalmanStateStorage(int dim) : a(dim), P(dim) {}
    Vector a;
    SpdMatrix P;
  };

  // Storage for the full Kalman filter and smoother, for scalar-valued time
  // series.
  struct ScalarKalmanStorage : public KalmanStateStorage {
    ScalarKalmanStorage() : ScalarKalmanStorage(0, true) {}
    explicit ScalarKalmanStorage(int dim, bool store_state_moments = true)
        : KalmanStateStorage(store_state_moments ? 0 : dim),
          K(dim),
          F(0),
          v(0)
    {}

    // Kalman gain
    Vector K;

    // Forward prediction variance.  Variance of y[t] given data to
    // time t-1.
    double F;

    // One step prediction error.  Difference between y[t] and its
    // prediction given data to time t-1.
    double v;
  };

  // Storage for the full Kalman filter and smoother, for multivariate time
  // series.
  struct MultivariateKalmanStorage : public KalmanStateStorage {
    MultivariateKalmanStorage() {}
    MultivariateKalmanStorage(int observation_dim, int state_dim,
                              bool store_state_moments)
        : KalmanStateStorage(store_state_moments ? state_dim : 0),
          kalman_gain_(state_dim, observation_dim),
          forecast_precision_(observation_dim),
          forecast_precision_log_determinant_(negative_infinity()),
          forecast_error_(observation_dim) {}

    // The Kalman gain K[t] shows up in the updating equation:
    //       a[t+1] = T[t] * a[t] + K[t] * v[t].
    // Rows correspond to states and columns to observation elements.
    Matrix kalman_gain_;

    // Inverse of Var(y[t] | Y[t-1]).
    SpdMatrix forecast_precision_;

    // The log determinant of forecast_precision_.
    double forecast_precision_log_determinant_;

    // y[t] - E(y[t] | Y[t-1]).  The dimension matches y[t], which might vary
    // across t.
    Vector forecast_error_;

    // Computed from the Durbin-Koopman disturbance smoother.  DK do a poor job
    // of explaining what r_ is, but it is a scaled version of the state
    // disturbance error (or something...).  It is produced by
    // smooth_disturbances_fast() and used by propagate_disturbances().
    Vector r_;
  };

}  // namespace BOOM

#endif  // BOOM_KALMAN_STORAGE_HPP

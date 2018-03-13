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

#ifndef BOOM_SCALAR_KALMAN_STORAGE_HPP
#define BOOM_SCALAR_KALMAN_STORAGE_HPP

#include <LinAlg/Vector.hpp>
#include <LinAlg/SpdMatrix.hpp>

namespace BOOM{

  // LightKalmanStorage is 'light' because it does not keep a copy of
  // 'a' (the state forecast/state value) or P (variance of state
  // forecast/value).  The struct uses the mathematical notation from
  // Durbin and Koopman (2001) to make it easy to follow the math of
  // the Kalman filter.
  struct LightKalmanStorage{
    // Kalman gain
    Vector K;

    // Forward prediction variance.  Variance of y[t] given data to
    // time t-1.
    double F;

    // One step prediction error.  Difference between y[t] and its
    // prediction given data to time t-1.
    double v;

    LightKalmanStorage(){}
    LightKalmanStorage(int dim) : K(dim) {}
  };

  struct ScalarKalmanStorage : public LightKalmanStorage{
    // Expected value of the state at time t given data to time t-1.
    Vector a;
    // Variance of state at time t given data to time t-1.
    SpdMatrix P;
    ScalarKalmanStorage() : LightKalmanStorage() {}
    ScalarKalmanStorage(int dim) : LightKalmanStorage(dim), a(dim), P(dim) {}
  };
}  // namespace BOOM

#endif// BOOM_SCALAR_KALMAN_STORAGE_HPP

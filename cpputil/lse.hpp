// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_LSE_HPP
#define BOOM_LSE_HPP

#include <cmath>
#include "LinAlg/Vector.hpp"

namespace BOOM {
  double lse(const Vector &v);
  double lse_safe(const Vector &v);
  double lse_fast(const Vector &v);

  // The log of the sum of 2 exponentials.  log(exp(x) + exp(y))
  inline double lse2(double x, double y) {
    // returns log( exp(x) + exp(y));
    if (x < y) {
      double tmp(x);
      x = y;
      y = tmp;
    }
    return x + ::log1p(::exp(y - x));
  }

  // The log of the difference of 2 exponentials.  log(exp(x) -
  // exp(y)).  Must be called with x >= y.  If x == y then
  // negative_infinity() is returned.  If x < y then an exception is
  // thrown.
  double lde2(double x, double y);

}  // namespace BOOM
#endif  // BOOM_LSE_HPP

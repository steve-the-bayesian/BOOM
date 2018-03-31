// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef BOOM_CPP_MATH_UTILS_H
#define BOOM_CPP_MATH_UTILS_H

#include <cmath>
#include <limits>
#include "cpputil/portable_math.hpp"

namespace BOOM {
  inline int I(int r, int s) { return r == s ? 1 : 0; }
  double safelog(double x);

  constexpr double infinity() {
        return std::numeric_limits<double>::infinity(); 
  }

  constexpr double negative_infinity() {
    return -1 * std::numeric_limits<double>::infinity();
  }

  template <class T>
  inline T square(T x) {
    return x * x;
  }
  inline bool finite(double x) { return std::isfinite(x); }

  inline int divide_rounding_up(int a, int b) {
    int ans = a / b;
    return ans * b < a ? ans + 1 : ans;
  }

  using std::isnan;
  using std::log1p;
  using std::expm1;
  
}  // namespace BOOM

#endif  // BOOM_CPP_MATH_UTILS_H

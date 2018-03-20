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
#include "LinAlg/Vector.hpp"

#include <cmath>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  double lse_safe(const Vector &eta) {
    double m = eta.max();
    if (m == negative_infinity()) return m;
    double tmp = 0;
    uint n = eta.size();
    for (uint i = 0; i < n; ++i) tmp += exp(eta[i] - m);
    return m + log(tmp);
  }

  double lse_fast(const Vector &eta) {
    double ans = 0;
    uint n = eta.size();
    const double *d(eta.data());
    for (uint i = 0; i < n; ++i) {
      ans += exp(d[i]);
    }
    if (ans <= 0) {
      return negative_infinity();
    }
    return log(ans);
  }

  double lse(const Vector &eta) { return lse_safe(eta); }

  double lde2(double x, double y) {
    if (x <= y) {
      if (x < y) {
        report_error("x must be >= y in lde2");
      }
      return negative_infinity();
    }
    return x + log1p(-exp(y - x));
  }
}  // namespace BOOM

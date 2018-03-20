/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include <cmath>
#include "cpputil/Constants.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  // TODO(stevescott): This will work fine for small values of
  // dimension.  If dimension is large there might be more efficient
  // implementations.
  double lmultigamma(double x, int dimension) {
    if (dimension <= 0) {
      report_error("Dimension argument must be a positive integer.");
    }
    double ans = Constants::log_pi * dimension * (dimension - 1) / 4.0;
    for (int i = 1; i <= dimension; ++i) {
      ans += lgamma(x + (1 - i) / 2.0);
    }
    return ans;
  }

  // Can be more efficient than two calls to lmultigamma if extra is
  // small and dimension is large.
  double lmultigamma_ratio(double x, int extra, int dimension) {
    if (dimension <= 0) {
      report_error("Dimension argument must be a positive integer.");
    }
    if (extra == 0) {
      return 0;
    }
    if (extra < 0) {
      return -lmultigamma_ratio(x, -extra, dimension);
    }
    if (2 * extra >= dimension) {
      return lmultigamma(x + extra / 2.0, dimension)
          - lmultigamma(x, dimension);
    }
    double ans = 0;
    for (int i = 0; i < extra; ++i) {
      ans += lgamma(x +  (extra - i) / 2.0)
          - lgamma(x + (1 + i - dimension) / 2.0);
    }
    return ans;
  }

}  // namespace BOOM

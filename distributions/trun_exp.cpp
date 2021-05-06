// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  // Draw from the distribution proportional to exp(slope * x) between
  // lo and hi.  The CDF is
  //
  //   F(x) = (exp(slope * x) - exp(slope * lo))
  //            / (exp(slope * hi) - exp(slope * lo))
  //
  // Setting F(x) = u and solving for x gives the draw, so if u ~ U(0, 1)
  //
  //   x = log( u * exp(slope * b) + (1-u) * exp(slope * a)) / slope
  //
  // This function should handle infinite limits gracefully.
  double rpiecewise_log_linear_mt(RNG &rng, double slope, double lo,
                                  double hi) {
    // First handle all the strange cases.
    if (fabs(hi - lo) < 1e-7) {
      return lo;
    } else if (lo > hi) {
      report_error("Limits are reversed in rpiecewise_log_linear_mt.");
    } else if ((lo == negative_infinity() && slope <= 0) ||
               (hi == infinity() && slope >= 0)) {
      report_error(
          "slope is incompatible with infinite limits in"
          "rpiecewise_log_linear_mt");
    } else if (lo == negative_infinity()) {
      return hi - rexp_mt(rng, slope);
    } else if (hi == infinity()) {
      return lo + rexp_mt(rng, -slope);
    }

    // Now we know that lo and hi are both finite, with lo < hi.
    double u = 0.0;
    constexpr double eps = std::numeric_limits<double>::min();
    while (u < eps || u >= 1.0 - eps) {
      u = runif_mt(rng, 0, 1);
    }

    double first_part = log(u) + slope * hi;
    double second_part = log(1 - u) + slope * lo;
    return lse2(first_part, second_part) / slope;
  }

  double rtrun_exp_mt(RNG &rng, double lam, double lo, double hi) {
    return rpiecewise_log_linear_mt(rng, -lam, lo, hi);
  }

  double rtrun_exp(double lam, double lo, double hi) {
    return rtrun_exp_mt(GlobalRng::rng, lam, lo, hi);
  }
}  // namespace BOOM

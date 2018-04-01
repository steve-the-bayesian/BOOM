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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#include <cmath>
#include <sstream>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

/* functions for the triangular distribution over the interval (x0,
   x1) with breakpoint xm */

namespace BOOM {
  double rtriangle(double x0, double x1, double xm) {
    return rtriangle_mt(GlobalRng::rng, x0, x1, xm);
  }

  double rtriangle_mt(RNG& rng, double x0, double x1, double xm) {
    /* simulates from the noncentral triangle distribution on the
       interval (x0, x1) with a break at xm.  If xm < x0 || xm > x1 then
       xm is taken to be the midpoint of the interval.
    */

    double y, m0, m1, a0, u;

    if (x1 < x0) {
      std::ostringstream err;
      err << "error in rtriangle_mt: called with" << std::endl
          << "x0 = " << x0 << std::endl
          << "x1 = " << x1 << std::endl
          << "xm = " << xm << std::endl
          << "x0 must be less than x1";
      report_error(err.str());
    } else if (x0 == x1)
      return x0;

    if (xm < x0 || xm > x1) xm = (x0 + x1) / 2.0;

    y = 2.0 / (x1 - x0);
    m0 = y / (xm - x0);
    m1 = y / (xm - x1);

    a0 = 0.5 * y * (xm - x0);
    u = runif_mt(rng, 0, 1);

    double ans = 0;
    if (!std::isfinite(a0)) {
      report_error("an unknown error occurred in rtriangle_mt");
    }
    
    if (u < a0) {
      ans = x0 + sqrt(2 * u / m0); /* area of left right triangle */
    } else {
      ans = x1 - sqrt(-2.0 * (1 - u) / m1); /* area of right right triangle */
    }
    return ans;
  }
  /*======================================================================*/
  double dtriangle(double x, double x0, double x1, double xm, bool logscale) {
    double m0, m1, y, ans;

    if (x1 < x0) {
      std::ostringstream err;
      err << "error in dtriangle: called with" << std::endl
          << "x0 = " << x0 << std::endl
          << "x1 = " << x1 << std::endl
          << "xm = " << xm << std::endl
          << "logscale = " << logscale << std::endl
          << "x0 must be less than x1";

      report_error(err.str());
    }
    if (x0 == x1) return x0;

    if (x < x0 || x > x1) return (logscale ? negative_infinity() : 0);

    if (xm < x0 || xm > x1) xm = (x0 + x1) / 2.0;
    y = 2.0 / (x1 - x0);
    m0 = y / (xm - x0);
    m1 = y / (xm - x1);
    ans = (x < xm ? m0 * (x - x0) : m1 * (x - x1));
    return (logscale ? log(ans) : ans);
  }
  /*======================================================================*/
  double ptriangle(double x, double x0, double x1, double xm, bool lower_tail) {
    double y;

    if (x1 < x0) {
      std::ostringstream err;
      err << "error in ptriangle: called with" << std::endl
          << "x0 = " << x0 << std::endl
          << "x1 = " << x1 << std::endl
          << "xm = " << xm << std::endl
          << "x0 must be less than x1";
      report_error(err.str());
    } else if (x0 == x1)
      return x0;

    if (x < x0) return lower_tail ? 0.0 : 1.0;
    if (x > x1) return lower_tail ? 1.0 : 0.0;

    if (xm < x0 || xm > x1) xm = (x0 + x1) / 2.0;
    y = 2.0 / (x1 - x0);

    double ans = 0;
    if (!std::isfinite(x) || !std::isfinite(xm)) {
      report_error("Non-finite inputs to ptriangle.");
    } else if (x <= xm) {
      x -= x0;
      double m0 = y / (xm - x0);
      double a0 = 0.5 * m0 * x * x;
      ans = lower_tail ? a0 : 1 - a0;
    } else {
      double m1 = y / (xm - x1);
      double b = 0.5 * m1 * (x - x1) * (x1 - x);
      ans = lower_tail ? b : 1 - b;
    }
    return ans;
  }
  /*======================================================================*/
  double qtriangle(double p, double x0, double x1, double xm) {
    double y, m0, m1, a0;

    if (x1 < x0) {
      std::ostringstream err;
      err << "error in qtriangle: called with" << std::endl
          << "x0 = " << x0 << std::endl
          << "x1 = " << x1 << std::endl
          << "xm = " << xm << std::endl
          << "x0 must be less than x1";
      report_error(err.str());
    } else if (x0 == x1)
      return x0;

    if (xm < x0 || xm > x1) xm = (x0 + x1) / 2.0;

    y = 2.0 / (x1 - x0);
    m0 = y / (xm - x0);
    m1 = y / (xm - x1);

    a0 = 0.5 * y * (xm - x0);

    double ans = 0;
    if (!std::isfinite(p)) {
      report_error("Non finite value for p in qtriangle.");
    } else if (!std::isfinite(a0)) {
      report_error("Nonfinite value for a0 in qtriangle.");
    } else if (p < a0) {
      ans = x0 + sqrt(2 * p / m0); /* area of left right triangle */
    } else {
      ans = x1 - sqrt(-2.0 * (1 - p) / m1); /* area of right right triangle */
    }
    return ans;
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "distributions.hpp"

#include <cmath>
#include <sstream>

#include "uint.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  double dusp(double x, double z0, bool logscale) {
    if (z0 <= 0) {
      report_error("non-positive z0 in  dusp");
    }
    if (x <= 0) {
      return logscale ? negative_infinity() : 0.0;
    }

    if (logscale) {
      return log(z0) - 2.0 * log(z0 + x);
    } else {
      double zpx = z0 + x;
      return z0 / square(zpx);
    }
  }
  //======================================================================
  double pusp(double x, double z0, bool logscale) {
    if (x <= 0) return logscale ? BOOM::negative_infinity() : 0;
    if (z0 <= 0) {
      report_error("error: non-positive z0 in  pusp");
    }
    if (logscale)
      return log(x) - log(x + z0);
    else
      return x / (x + z0);
  }
  //======================================================================
  double qusp(double p, double z0) {
    if (z0 <= 0) {
      ostringstream msg;
      msg << "error: non-positive z0 in qusp:  z0 = " << z0;
      report_error(msg.str());
    }

    if (p <= 0 || p >= 1) {
      ostringstream msg;
      msg << "probability out of range in qusp: p = " << p;
      report_error(msg.str());
    }
    return z0 * p / (1 - p);
  }
  //======================================================================
  double rusp(double z0) { return qusp(runif(0, 1), z0); }
  double rusp_mt(RNG &rng, double z0) { return qusp(runif_mt(rng, 0, 1), z0); }
}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
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

#ifndef BOOM_KS_CRITICAL_VALUES_HPP_
#define BOOM_KS_CRITICAL_VALUES_HPP_
#include <cmath>
#include "cpputil/report_error.hpp"

namespace BOOM {
  // Critical value for the Kolmogorov Smirnoff test statistic
  // max(abs(ECDF(y) - F(y))), where ECDF(y) is the empirical CDF of a
  // continuous random variable y with distribution function F.
  //
  // This function is valid only for n >= 35.
  //
  // Args:
  //   n:  sample size on which the KS test is based.
  //   alpha:  Significance level for the test.
  //
  // Returns:
  //   A critical value K such that if the KS test statistic exceeds K
  //   the KS test is rejected at significance level alpha.
  double ks_critical_value(double n, double alpha = .05) {
    if (n < 35) {
      report_error("ks_critical_value is only valid for n >=35.");
    }
    if (alpha <= 0 || alpha >= 1) {
      report_error("alpha must be between 0 and 1.");
    }
    return sqrt(-.5 * log(alpha / 2) / n);
  }
}  // namespace BOOM
#endif  //  BOOM_KS_CRITICAL_VALUES_HPP_

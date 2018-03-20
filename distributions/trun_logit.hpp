// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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
#ifndef BOOM_DISTRIBUTIONS_TRUN_LOGIT_HPP_
#define BOOM_DISTRIBUTIONS_TRUN_LOGIT_HPP_

#include "distributions/rng.hpp"
#include "distributions/trun_logit.hpp"

namespace BOOM {

  // Let F(z) denote the logit CDF, and f(z) the pdf.  Note that f(z) =
  // F(z) * (1 - F(z)).
  //
  // To compute the mean we need to integrate z * f(z) from x to
  // infinity.  The anti-derivative of z*f(z) is A(z) = (z * F) -
  // log(exp(z) + 1).  A(infinity) = 0, and A(-infinity) = 0.  Thus
  //
  // E(z | z > x) = A(infinity) - A(x) = -A(x)
  //
  // and E(z | z < x) = A(x) - A(-infinity) = A(x)
  //
  // Args:
  //   truncation_point:  The point at which the distribution is truncated.
  //   above: If true the support of the distribution is above
  //     truncation_point.  If false the support is below
  //     truncation_point.
  // Returns:
  //   The expected value of a standard logit deviate, conditional on
  //   truncation.
  double trun_logit_mean(double truncation_point, bool above);
  inline double trun_logit_mean(double mean, double truncation_point,
                                bool above) {
    return trun_logit_mean(truncation_point - mean, above) + mean;
  }

  // To compute the variance of the truncated logit, we compute E(z^2)
  // and then subtract E(z)^2.
  //
  // Let f(z) be the pdf of the standard logistic distribution.  The
  // anti-derivative of z^2 * f(z) is
  //
  // B(z) = z * (z * F(z) - 2 * log(1 + exp(z))) - 2 * Li_2(-exp(z))
  //
  // where Li_2 is the polylogarithm function (dilog).
  double trun_logit_variance(double truncation_point, bool above);
  inline double trun_logit_variance(double mean, double truncation_point,
                                    bool above) {
    return trun_logit_variance(truncation_point - mean, above);
  }

  // A random draw from the truncated logistic distribution.
  // Args:
  //   rng:  The random number generator.
  //   mean:  The mean of the logistic distribution, before truncation.
  //   cutpoint:  The point of truncation.
  //   above: If true, then support is above 'cutpoint'.  Otherwise
  //     support is below 'cutpoint.'
  double rtrun_logit_mt(RNG &rng, double mean, double cutpoint, bool above);
}  // namespace BOOM

#endif  //  BOOM_DISTRIBUTIONS_TRUN_LOGIT_HPP_

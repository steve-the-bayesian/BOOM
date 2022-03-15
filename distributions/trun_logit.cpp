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

#include "distributions/trun_logit.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/Constants.hpp"
#include "distributions.hpp"           // for plogis
#include "math/special_functions.hpp"  // for dilog
#include "stats/logit.hpp"             // for lope

namespace BOOM {

  namespace {

    inline double dilog_negative_exp_approx(double x) {
      if (x > 10) {
        // Approximates dilog(-exp(x)) for large x.  We use the expression
        // below with z = -exp(x).  If x is very large then -1/exp(x) is
        // close to zero and dilog(-1/exp(x)) = 0.
        //
        //  dilog(z) + dilog(1/z) = -pi^2/6 - .5 * (log(-z))^2
        //
        // If x > 10 then the approximation has absolute error less that 5e-5.
        return -Constants::pi_squared_over_6 - .5 * x * x;
      } else if (x > -10) {
        return dilog(-exp(x));
      } else {
        return 0;
      }
    }

    // Returns the second moment of the logistic distribution.  The
    // anti-derivative of (z^2 f(z)) is z * (z * F(z) - 2 * log(1 +
    // exp(z))) - 2 * Li_2(-exp(z)), where Li_2 is the polylog function
    // of order 2 (aka the dilog function).
    //
    // The
    inline double second_moment_antiderivative(double x) {
      if (x == infinity()) {
        // Values at infinity and negative_infinity courtesy of Wolfram.
        return Constants::pi_squared_over_3;
      } else if (x == negative_infinity()) {
        return 0;
      }
      double ans = x * (plogis(x) * x - 2 * lope(x));
      if (x > 10) {
        ans -= 2 * dilog_negative_exp_approx(x);
      } else if (x > -10) {
        ans -= 2 * dilog(-exp(x));
      } else {
        // x < -10 then exp(x) is effectively zero, so dilog(-exp(x)) is
        // very close to zero, and can be ignored.
        ans -= 0;
      }
      return ans;
    }

  }  // namespace

  // TODO: Check this against simulations.

  // Let G(x) denote the antiderivative of x * f(x), where f(x) is the
  // logistic density, and let F(x) denote the logistic cumulative
  // distribution function.  G(x) = x * F(x) - log(1 + exp(x)).  One can
  // show that G(infinity) = G(-infinity) = 0.
  //
  // Case I: Upper truncation: The expectation of a truncated logistic
  // variate Z, given Z > x, is
  //
  // (G(infinity) - G(x)) / (1 - F(x)) = -G(x) * (1 + exp(x))
  //
  // One can simplify this to
  //
  // log(1 + exp(x)) + exp(x) * log(1 + exp(-x)).
  //
  // As x -> infinity this converges to x + 1.  As x ->
  // -infinity this converges to 0.
  //
  // Case II: Lower truncation: The expectation of a truncated logistic
  // variable Z given Z < x is
  //
  // (G(x) - G(-infinity)) / F(x) = G(x) / F(x)
  //  x - log(1 + exp(x)) * (1 + exp(-x))
  //
  // The limit as x -> -infinity is x - 1.  The limit as x -> infinity
  // is 0.
  //
  // For the purpose of this function, infinity is around 20.
  double trun_logit_mean(double truncation_point, bool above) {
    if (truncation_point == BOOM::infinity()) {
      return above ? truncation_point : 0;
    } else if (truncation_point == BOOM::negative_infinity()) {
      return above ? 0 : truncation_point;
    }
    if (above) {
      // Support is above truncation_point.
      if (truncation_point > 20) {
        return truncation_point + 1;
      } else {
        return lope(truncation_point) +
               exp(truncation_point) * lope(-truncation_point);
      }
    } else {
      // Support is below truncation_point.
      if (truncation_point < -20) {
        return truncation_point - 1;
      } else {
        return truncation_point -
               lope(truncation_point) * (1 + exp(-truncation_point));
      }
    }
  }

  // Computes the variance of the standard logistic distribution
  // truncated at 'truncation_point.'
  //
  // Args:
  //   truncation_point:  The point of truncation.
  //   above: If true, then the support of the distribution is above
  //     truncation_point.  Otherwise the support is below
  //     truncation_point.
  double trun_logit_variance(double truncation_point, bool above) {
    if (!above) {
      return trun_logit_variance(-truncation_point, true);
    }
    double ans = 0;
    double probability_below = plogis(truncation_point);
    ans = second_moment_antiderivative(BOOM::infinity()) -
        second_moment_antiderivative(truncation_point);
    ans /= (1 - probability_below);
    ans -= square(trun_logit_mean(truncation_point, above));
    return ans;
  }

  // Sample from the truncated logistic distribution.
  //
  // Args:
  //   rng:  A random number generator.
  //   mean: The mean of the untruncated logistic distribution to be
  //     sampled.
  //   cutpoint:  The point of truncation.
  //   above: If true, the support of the distribution to be sampled is
  //     above the cutpoint.  If false it is below the cutpoint.
  //
  // Returns:
  //   A draw from the truncated logistic distribution.
  double rtrun_logit_mt(RNG &rng, double mean, double cutpoint, bool above) {
    // Draw from a standard logit truncated at curpoint - mean, and then
    // add in the mean.
    double cutpoint_prob = plogis(cutpoint - mean);
    double ans = 0;
    if (above) {
      ans = runif_mt(rng, cutpoint_prob, 1);
    } else {
      ans = runif_mt(rng, 0, cutpoint_prob);
    }
    return qlogis(ans) + mean;
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#include "distributions/inverse_gaussian.hpp"
#include <cmath>
#include <stdexcept>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  double dig(double x, double mu, double lambda, bool logscale) {
    const double log_two_pi(1.83787706640935);
    if (x <= 0) return logscale ? negative_infinity() : 0;
    if (mu <= 0) {
      report_error("mu <= 0 in dig");
    }
    if (lambda <= 0) {
      report_error("lambda <= 0 in dig");
    }

    double ans = -lambda * pow(x - mu, 2) / (2 * mu * mu * x);
    ans += .5 * (log(lambda) - log_two_pi - 3 * log(x));
    return logscale ? ans : exp(ans);
  }

  double pig(double x, double mu, double lambda, bool logscale) {
    if (x <= 0) return logscale ? negative_infinity() : 0;
    if (mu <= 0) {
      report_error("mu <= 0 in pig");
    }
    if (lambda <= 0) {
      report_error("lambda <= 0 in pig");
    }

    double rlx = sqrt(lambda / x);
    double xmu = x / mu;
    double ans =
        pnorm(rlx * (xmu - 1)) + exp(2 * lambda / mu) * pnorm(-rlx * (xmu + 1));
    return logscale ? log(ans) : ans;
  }

  double rig_mt(RNG& rng, double mu, double lambda) {
    double y = rnorm_mt(rng);
    y = y * y;
    double mu2 = mu * mu;
    double muy = mu * y;
    double mu2lam = .5 * mu / lambda;
    double x = mu + muy * mu2lam - mu2lam * sqrt(muy * (4 * lambda + muy));
    double z = runif_mt(rng);
    if (z > mu / (mu + x)) return mu2 / x;
    return x;
  }
}  // namespace BOOM

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

#include <cmath>
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"  // rnorm, etc

#include "distributions/Rmath_dist.hpp"

namespace BOOM {

  double dstudent(double x, double mu, double sigma, double df, bool logscale) {
    /* univariate student density where x = z/w + mu, where z~N(0,
       sigsq) and w^2~Gamma(df/2, df/2) (i.e. w^2 has been divided by
       its degrees of freedom) */

    double ans;
    if (sigma == 0) return (x == mu ? infinity() : 0.0);

    ans = (x - mu) / sigma;
    ans = dt(ans, df, 1) - log(sigma); /* log sigma from the jacobian */
    if (logscale)
      return ans;
    else
      return exp(ans);
  }

  double pstudent(double x, double mu, double sigma, double df,
                  bool lower_tail, bool logscale) {
    return pt((x - mu) / sigma, df, lower_tail, logscale);
  }

  double qstudent(double p, double mu, double sigma, double df,
                  bool lower_tail, bool logscale) {
    return qt(p, lower_tail, logscale) * sigma + mu;
  }

  /*======================================================================*/

  double rstudent(double mu, double sigma, double df) {
    return rstudent_mt(GlobalRng::rng, mu, sigma, df);
  }

  double rstudent_mt(RNG& rng, double mu, double sigma, double df) {
    double w = rgamma_mt(rng, df / 2.0, df / 2.0);
    return rnorm_mt(rng, mu, sigma / sqrt(w));
  }
}  // namespace BOOM

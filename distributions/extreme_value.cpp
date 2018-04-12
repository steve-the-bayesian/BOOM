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
#include <cassert>
#include <cmath>
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    const double euler_gamma = -0.577215664901533;
  }

  // Pr(Z <= z) = Pr(exp(-Z) >= exp(-z))
  double pexv(double x, double mu, double sigma, bool logscale) {
    double u =  (x - mu) / sigma;
    double y = exp(-u);
    if (logscale) {
      return -y;
    } else {
      return exp(-y);
    }
  }
  
  double dexv(double x, double mu, double sigma, bool logscale) {
    // Density of the extreme value distribution with centrality parameter 'mu'
    // and variance sigma^2 * pi^2/6.

    // f(eps | mu, sigma) = (1/sigma) exp( -(z - exp(-z)))
    // where z = (eps - mu) / sigma
    
    if (sigma <= 0) {
      report_error("sigma must be positive in dexv.");
    }
    double z = (x - mu) / sigma;
    double ans = -z - exp(-z) - log(sigma);
    return logscale ? ans : exp(ans);
  }

  double rexv(double mu, double sigma) {
    return rexv_mt(GlobalRng::rng, mu, sigma);
  }

  double rexv_mt(RNG& rng, double mu, double sigma) {
    if (sigma == 0.0) {
      return mu + euler_gamma;
    } else if (sigma < 0) {
      report_error("Sigma must be non-negative in rexv_mt.");
    }
    return -log(rexp_mt(rng, 1.0)) * sigma + mu;
  }
}  // namespace BOOM

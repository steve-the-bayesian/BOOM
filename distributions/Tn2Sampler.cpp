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

#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  Tn2Sampler::Tn2Sampler(double lo, double hi)
      : x(2), logf(2), dlogf(2), knots(3), cdf(2) {
    x[0] = lo;
    x[1] = hi;
    logf[0] = f(lo);
    logf[1] = f(hi);
    dlogf[0] = df(lo);
    dlogf[1] = df(hi);
    refresh_knots();
    update_cdf();
  }

  double Tn2Sampler::f(double x) const { return -.5 * x * x; }

  double Tn2Sampler::df(double x) const { return -x; }

  double Tn2Sampler::hull(double z, uint k) const {
    double xk = x[k];
    double dk = dlogf[k];
    double yk = logf[k];
    return yk + dk * (z - xk);
  }

  double Tn2Sampler::compute_knot(uint k) const {
    // returns the location of the intersection of the tanget line at
    // x[k] and x[k-1]
    double y2 = logf[k];
    double y1 = logf[k - 1];
    double d2 = dlogf[k];
    double d1 = dlogf[k - 1];
    double x2 = x[k];
    double x1 = x[k - 1];

    double ans = (y1 - d1 * x1) - (y2 - d2 * x2);
    ans /= (d2 - d1);
    return ans;
  }

  void Tn2Sampler::add_point(double z) {
    if (z > x.back()) {
      report_error("z out of bounds (too large) in Tn2Sampler::add_point");
    }
    if (z < x[0]) {
      report_error("z out of bounds (too small) in Tn2Sampler::add_point");
    }
    IT it = std::lower_bound(x.begin(), x.end(), z);
    int k = it - x.begin();
    x.insert(it, z);
    logf.insert(logf.begin() + k, f(z));
    dlogf.insert(dlogf.begin() + k, df(z));
    refresh_knots();
    update_cdf();
  }

  // wasteful!  should only update a knot between the x's, but adding
  // an x will change two knots
  void Tn2Sampler::refresh_knots() {
    knots.resize(x.size() + 1);
    knots[0] = x[0];
    knots.back() = x.back();
    for (uint i = 1; i < knots.size() - 1; ++i) knots[i] = compute_knot(i);
  }

  void Tn2Sampler::update_cdf() {
    // cdf[i] is the integral of the outer hull from knots[i] to
    // knots[i+1]
    uint n = x.size();
    cdf.resize(n);
    double y0 = logf[0];
    for (uint k = 0; k < n; ++k) {
      double d = dlogf[k];
      double y = hull(knots[k], k);
      double increment;
      if (fabs(d) < .00000000001) {
        increment = exp(y - y0) * (knots[k + 1] - knots[k]);
      } else {
        increment = (exp(y - y0) / d) * expm1(d * knots[k + 1] - knots[k]);
      }
      cdf[k] = (k == 0 ? increment : cdf[k - 1] + increment);
    }
  }

  double Tn2Sampler::draw(RNG &rng) {
    double u = runif_mt(rng, 0, cdf.back());
    IT pos = std::lower_bound(cdf.begin(), cdf.end(), u);
    uint k = pos - cdf.begin();
    // draw from the doubly truncated exponential distribution
    double lo = knots[k];
    double hi = knots[k + 1];
    double lam = -1 * dlogf[k];
    double cand;
    if (lam == 0 ||
        fabs(hi - lo) < std::sqrt(std::numeric_limits<double>::epsilon())) {
      cand = runif_mt(rng, lo, hi);
    } else {
      cand = rtrun_exp_mt(rng, lam, lo, hi);
    }
    double target = f(cand);
    double logu = hull(cand, k) - rexp_mt(rng, 1);
    if (logu < target) return cand;
    add_point(cand);
    return draw(rng);
  }
}  // namespace BOOM

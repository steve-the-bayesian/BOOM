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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  double trun_norm(double a) { return trun_norm_mt(GlobalRng::rng, a); }

  double rtrun_norm(double mu, double sigma, double a, bool gt) {
    return rtrun_norm_mt(GlobalRng::rng, mu, sigma, a, gt);
  }

  double rtrun_norm_mt(RNG &rng, double mu, double sigma, double a, bool gt) {
    /* draws a truncated normal deviate from the normal distribution
       with mean mu, sd=sigma.  The density is truncated such that z > a
       if gt==1 and z<a if gt==0. */
    double x;
    if (gt) {
      x = mu + sigma * trun_norm_mt(rng, (a - mu) / sigma);
    } else {
      x = mu - sigma * trun_norm_mt(rng, (mu - a) / sigma);
    }
    return x;
  }
  //=======================================================================
  double dtrun_norm(double x, double mu, double sig, double cut, bool below,
                    bool logscale) {
    /*  if(below) this function returns p(x | x<cut), otherwise it
       returns p(x|x>cut), where x~N(mu, sig^2).  if(logscale) the
       answer is returned on the log scale, otherwise it is returned on
       the probability scale */
    double ans;
    ans = dnorm(x, mu, sig, 1); /* on log scale */
    ans -= pnorm(cut, mu, sig, below, 1);
    return logscale ? ans : exp(ans);
  }
  /*======================================================================*/
  double dtrun_norm_2(double x, double mu, double sig, double lo, double hi,
                      bool logscale) {
    /* returns p(x | lo < x < hi) where x~N(mu, sig^2) */
    double ans, nc;
    if (hi < lo)
      ans = negative_infinity();
    else if (hi == lo)
      ans = (x == hi ? infinity() : negative_infinity());
    else {
      ans = dnorm(x, mu, sig, 1); /* answer on log scale */
      nc = pnorm(hi, mu, sig, 1, 0) - pnorm(lo, mu, sig, 1, 0);
      nc = log(nc);
      ans -= nc;
    }
    return logscale ? ans : exp(ans);
  }

  /*======================================================================*/
  inline double dnorm_inv(double logp, double mu, double sig) {
    const double log_root_2pi_inv = -0.918938533204673;

    double tmp = logp + log(sig) - log_root_2pi_inv;
    tmp *= -2 * sig * sig;
    return mu + sqrt(tmp);
  }

  inline double reflect(double x, double mu) {
    double dist = fabs(x - mu);
    return x < mu ? mu + dist : mu - dist;
  }

  //======================================================================
  namespace {
    std::ostream &operator<<(std::ostream &out, const std::vector<double> &v) {
      for (uint i = 0; i < v.size(); ++i) out << v[i] << " ";
      return out;
    }
  }  // namespace
  //----------------------------------------------------------------------
  std::ostream &TnSampler::print(std::ostream &out) const {
    using std::endl;
    out << "x     = " << x << std::endl
        << "logf  = " << logf << std::endl
        << "dlogf = " << dlogf << std::endl
        << "knots = " << knots << std::endl
        << "cdf   = " << cdf << std::endl
        << std::endl;
    return out;
  }
  //----------------------------------------------------------------------
  TnSampler::TnSampler(double a)
      : x(1, a), logf(1, f(a)), dlogf(1, df(a)), knots(1, a) {
    //    if(a < 1.0) add_point(1.0);
    // else update_cdf();
    update_cdf();
  }
  //----------------------------------------------------------------------
  void TnSampler::add_point(double z) {
    //  cout << "about to add a point: " << endl;
    //  this->print(cout);

    IT it = std::lower_bound(knots.begin(), knots.end(), z);

    if (it == knots.end()) {
      //    cout << "inserting at end " << endl;
      x.push_back(z);
      logf.push_back(f(z));
      dlogf.push_back(df(z));
    } else {
      //    cout << "inserting " << z << " before element " << *it << endl;
      uint k = it - knots.begin();
      x.insert(x.begin() + k, z);
      logf.insert(logf.begin() + k, f(z));
      dlogf.insert(dlogf.begin() + k, df(z));
    }

    refresh_knots();
    update_cdf();
  }
  //----------------------------------------------------------------------
  void TnSampler::refresh_knots() {
    // wasteful!  should only update a knot between the x's, but
    // adding an x will change two knots

    knots.resize(x.size());
    knots[0] = x[0];
    for (uint i = 1; i < knots.size(); ++i) knots[i] = compute_knot(i);
  }
  //----------------------------------------------------------------------
  double TnSampler::compute_knot(uint k) const {
    // returns the location of the intersection of the tanget line at
    // x[k] and x[k-1]

    if (k == 0) return x[0];
    double x2 = x[k];
    double y2 = logf[k];
    double d2 = dlogf[k];

    double x1 = x[k - 1];
    double y1 = logf[k - 1];
    double d1 = dlogf[k - 1];

    double ans = (y1 - d1 * x1) - (y2 - d2 * x2);
    ans /= (d2 - d1);
    return ans;
  }
  //----------------------------------------------------------------------
  double TnSampler::f(double x) const { return -.5 * x * x; }
  //----------------------------------------------------------------------
  double TnSampler::df(double x) const { return -x; }
  //----------------------------------------------------------------------

  // integral from [lo..hi] of exp(slope * x + intercept)
  inline double integral(double lo, double hi, double slope, double intercept) {
    return (1.0 / slope) *
           (exp(intercept + slope * hi) - exp(intercept + slope * lo));
  }

  void TnSampler::update_cdf() {
    // cdf[i] is the integral of the outer hull from knots[i] to
    // knots[i+1], where the last value is implicitly infinity.

    // cdf is un-normalized, so we divide everything by exp(y0)
    uint n = knots.size();
    cdf.resize(n);
    double y0 = logf[0];
    double last = 0;
    for (uint k = 0; k < knots.size(); ++k) {
      double d = dlogf[k];
      double y = logf[k] - y0;
      double z = x[k];
      double dinv = 1.0 / d;
      double inc1 = k == n - 1 ? 0 : dinv * exp(y - d * z + d * knots[k + 1]);
      double inc2 = dinv * exp(y - d * z + d * knots[k]);
      cdf[k] = last + inc1 - inc2;
      last = cdf[k];
    }
  }

  //----------------------------------------------------------------------
  double TnSampler::h(double z, uint k) const {
    double xk = x[k];
    double dk = dlogf[k];
    double yk = logf[k];
    return yk + dk * (z - xk);
  }
  //----------------------------------------------------------------------
  double TnSampler::draw(RNG &rng) {
    double u = runif_mt(rng, 0, cdf.back());
    IT pos = std::lower_bound(cdf.begin(), cdf.end(), u);
    uint k = pos - cdf.begin();
    double cand;
    if (k + 1 == cdf.size()) {
      // one sided draw..................
      cand = knots.back() + rexp_mt(rng, -1 * dlogf.back());
    } else {
      // draw from the doubly truncated exponential distribution
      double lo = knots[k];
      double hi = knots[k + 1];
      double lam = -1 * dlogf[k];
      cand = rtrun_exp_mt(rng, lam, lo, hi);
    }
    double target = f(cand);
    double hull = h(cand, k);
    double logu = hull - rexp_mt(rng, 1);
    if (logu < target) return cand;
    add_point(cand);
    return draw(rng);
  }
  //----------------------------------------------------------------------
  double trun_norm_mt(RNG &rng, double a) {
    if (a <= 0) {  // expect 1 rejection, with sd < sqrt(2)
      while (1) {
        double x = rnorm_mt(rng, 0, 1);
        if (x > a) return x;
      }
    }

    TnSampler sam(a);
    return sam.draw(rng);
  }

  void trun_norm_moments(double mu, double sigma, double cutpoint,
                         bool positive_support, double *mean,
                         double *variance) {
    double sigsq = sigma * sigma;
    if (positive_support) {
      double alpha = (cutpoint - mu) / sigma;
      // phi_ratio is phi(alpha) / (1 - Phi(alpha)).
      double phi_ratio =
          exp(dnorm(alpha, 0, 1, true) - pnorm(alpha, 0, 1, false, true));
      *mean = mu + sigma * phi_ratio;
      double delta = phi_ratio * (phi_ratio - alpha);
      *variance = sigsq * (1 - delta);
      if (*variance < 0) {
        *variance = 0;
      }
    } else {
      double beta = (cutpoint - mu) / sigma;
      // phi_ratio is phi(beta) / Phi(beta)
      double phi_ratio =
          exp(dnorm(beta, 0, 1, true) - pnorm(beta, 0, 1, true, true));
      *mean = mu - sigma * phi_ratio;
      *variance = sigsq * (1 - beta * phi_ratio - square(phi_ratio));
      if (*variance < 0) {
        *variance = 0;
      }
    }
  }

  //======================================================================

  double rtrun_norm_2_mt(RNG &rng, double mu, double sigma, double lo,
                         double hi) {
    if (hi >= BOOM::infinity()) {
      return rtrun_norm_mt(rng, mu, sigma, lo, true);
    }

    if (lo <= BOOM::negative_infinity()) {
      return rtrun_norm_mt(rng, mu, sigma, hi, false);
    }

    if (lo < mu && hi > mu) {
      if ((hi - lo) / sigma > .5) {
        // rejection sampling with normal envelope
        double y = lo - 1;
        while (y < lo || y > hi) {
          y = rnorm_mt(rng, mu, sigma);
        }
        return y;
      } else {  // rejection sampling with uniform envelope
        double phi_mu = dnorm(mu, mu, sigma, true);
        double phi = phi_mu;
        double u = phi + 1;
        double y = 0;
        while (u > phi) {
          y = runif_mt(rng, lo, hi);
          phi = dnorm(y, mu, sigma, true);
          u = phi_mu - rexp_mt(rng, 1);
        }
        return y;
      }
    }

    hi = (hi - mu) / sigma;
    lo = (lo - mu) / sigma;

    if (hi < 0) {
      double y = rtrun_norm_2_mt(rng, 0, 1, -hi, -lo);
      return mu - sigma * y;
    }

    try {
      Tn2Sampler sam(lo, hi);
      double y = sam.draw(rng);
      return y * sigma + mu;
    } catch (std::exception &e) {
      std::ostringstream err;
      err << "rtrun_norm_2_mt caught an exception when called with arguments"
          << std::endl
          << "    mu = " << mu << std::endl
          << " sigma = " << sigma << std::endl
          << "    lo = " << lo << std::endl
          << "    hi = " << hi << std::endl
          << "The error message of the captured exception is " << std::endl
          << e.what() << std::endl;
      report_error(err.str());
    } catch (...) {
      report_error("caught unknown exception in rtrun_norm_2_mt");
    }
    return 0;  // The only way to get here is to fail in the try/catch block
  }

  double rtrun_norm_2(double mu, double sigma, double lo, double hi) {
    return rtrun_norm_2_mt(GlobalRng::rng, mu, sigma, lo, hi);
  }
}  // namespace BOOM

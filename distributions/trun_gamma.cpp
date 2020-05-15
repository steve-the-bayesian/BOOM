// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "distributions/trun_gamma.hpp"
#include <cmath>  // log
#include <sstream>
#include "cpputil/math_utils.hpp"  // infinity
#include "cpputil/report_error.hpp"
#include "distributions.hpp"  // rgamma, runif
#include "distributions/BoundedAdaptiveRejectionSampler.hpp"

namespace BOOM {

  double rtg_init(double x, double a, double b, double cut, double logpstar);
  double rtg_slice(RNG &, double x, double a, double b, double cut);

  double dtrun_gamma(double x, double a, double b, double cut,
                     bool logscale, bool normalize) {
    if (a < 0 || b < 0 || cut < 0 || x < cut) {
      return BOOM::negative_infinity();
    }

    double ans;
    if (normalize) {
      ans = dgamma(x, a, b, true) - pgamma(cut, a, b, false, true);
    } else {
      ans = (a - 1) * log(x) - b * x;
    }
    return logscale ? ans : exp(ans);
  }

  //----------------------------------------------------------------------
  class LogGammaDensity {
   public:
    LogGammaDensity(double a, double b, double cut) : a_(a), b_(b), cut_(cut) {}
    double operator()(double x) const {
      return dtrun_gamma(x, a_, b_, cut_, true, false);
    }

   private:
    double a_, b_, cut_;
  };

  class DLogGammaDensity {
   public:
    DLogGammaDensity(double a, double b) : a_(a), b_(b) {}
    double operator()(double x) const { return (a_ - 1) / x - b_; }

   private:
    double a_, b_;
  };

  double rtrun_gamma(double a, double b, double cut, unsigned n) {
    return rtrun_gamma_mt(GlobalRng::rng, a, b, cut, n);
  }

  double rtrun_gamma_mt(RNG &rng, double a, double b, double cut, unsigned nslice) {
    double mode = (a - 1) / b;
    double x = cut;
    if (cut < mode) {  // rejection sampling
      do {
        x = rgamma_mt(rng, a, b);
      } while (x < cut);
      return x;
    }
    if (a > 1) {
      try {
        BoundedAdaptiveRejectionSampler sam(cut, LogGammaDensity(a, b, cut),
                                            DLogGammaDensity(a, b));
        return sam.draw(rng);
      } catch (std::exception &e) {
        std::ostringstream err;
        err << "Caught exception with error message:  " << std::endl
            << e.what() << std::endl
            << "in call to rtrun_gamma_mt with " << std::endl
            << "  a = " << a << std::endl
            << "  b = " << b << std::endl
            << "cut = " << cut << std::endl
            << "  nslice = " << nslice << std::endl;
        report_error(err.str());
      } catch (...) {
        report_error("caught unknown exception in rtrun_gamma_mt");
      }
    } else {
      for (unsigned i = 0; i < nslice; ++i) {
        x = rtg_slice(rng, x, a, b, cut);
      }
    }
    return x;
  }

  //----------------------------------------------------------------------

  double rtg_init(double x, double a, double b, double cut, double logpstar) {
    /*
     * finds a value of x such that dtrun_gamma(x,a,b,cut,true,false) <
     * logpstar.  This function will only be called if cut > mode, in which case
     * dtrun_gamma is a decreasing function.
     */

    double f = dtrun_gamma(x, a, b, cut, true) - logpstar;
    double fprime = ((a - 1) / x) - b;
    int attempts = 0;
    int max_attempts = 1000;
    while (f > sqrt(std::numeric_limits<double>::epsilon())) {
      x -= f / fprime;
      f = dtrun_gamma(x, a, b, cut, true) - logpstar;
      fprime = ((a - 1) / cut) - b;
      if (++attempts > max_attempts) {
        break;
      }
    }
    return x;
  }
  //----------------------------------------------------------------------
  double rtg_slice(RNG &rng, double x, double a, double b, double cut) {
    double logpstar = dtrun_gamma(x, a, b, cut, true) - rexp_mt(rng, 1.0);
    double lo = cut;
    double hi = rtg_init(x, a, b, cut, logpstar);
    x = runif_mt(rng, lo, hi);
    int trials = 0;
    int max_trials = 1000;
    while (dtrun_gamma(x, a, b, cut, true) < logpstar) {
      hi = x;
      x = runif_mt(rng, lo, hi);
      if (++trials > max_trials) {
        return cut;
      }
    }
    return (x);
  }

}  // namespace BOOM

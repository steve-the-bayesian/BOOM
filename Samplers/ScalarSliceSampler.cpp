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

#include "Samplers/ScalarSliceSampler.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include "uint.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef ScalarSliceSampler SSS;

  SSS::ScalarSliceSampler(const Fun &F, bool Unimodal, double dx, RNG *rng)
      : ScalarSampler(rng),
        logf_(F),
        suggested_dx_(dx),
        min_dx_(-1),
        lo_set_manually_(false),
        hi_set_manually_(false),
        unimodal_(Unimodal),
        estimate_dx_(true) {}

  void SSS::set_suggested_dx(double dx) { suggested_dx_ = dx; }
  void SSS::set_min_dx(double dx) { min_dx_ = dx; }
  void SSS::estimate_dx(bool yn) { estimate_dx_ = yn; }

  void SSS::set_limits(double Lo, double Hi) {
    assert(Hi > Lo);
    set_lower_limit(Lo);
    set_upper_limit(Hi);
  }
  
  void SSS::set_lower_limit(double Lo) {
    if (std::isfinite(Lo)) {
      lo_ = lower_bound_ = Lo;
      lo_set_manually_ = true;
    } else {
      lo_set_manually_ = false;
    }
  }
  void SSS::set_upper_limit(double Hi) {
    if (std::isfinite(Hi)) {
      hi_ = upper_bound_ = Hi;
      hi_set_manually_ = true;
    } else {
      hi_set_manually_ = false;
    }
  }

  void SSS::unset_limits() { hi_set_manually_ = lo_set_manually_ = false; }

  double SSS::logp(double x) const { return logf_(x); }

  double SSS::draw(double x) {
    find_limits(x);
    double logp_cand = 0;
    int number_of_tries = 0;
    do {
      double x_cand = runif_mt(rng(), lo_, hi_);
      logp_cand = logf_(x_cand);
      if (logp_cand < logp_slice_) {
        contract(x, x_cand, logp_cand);
        ++number_of_tries;
      } else
        return x_cand;
      if (number_of_tries > 100) {
        ostringstream err;
        err << "number of tries exceeded.  candidate value is " << x_cand
            << " with logp_cand = " << logp_cand << endl;
        handle_error(err.str(), x);
      }
    } while (logp_cand < logp_slice_);
    handle_error("should never get here", x);
    return 0;
  }

  // If a candidate draw winds up out of the slice then the
  // pseudo-slice can be made narrower to increase the chance of
  // success next time.  See Neal (2003).
  void SSS::contract(double x, double x_cand, double logp) {
    if (x_cand > x) {
      hi_ = x_cand;
      logphi_ = logp;
    } else {
      lo_ = x_cand;
      logplo_ = logp;
    }
    if (estimate_dx_) {
      suggested_dx_ = hi_ - lo_;
      if (suggested_dx_ < min_dx_) suggested_dx_ = min_dx_;
    }
  }

  // Driver function to find the limits of a slice containing 'x'.
  // Logic varies according to whether the distribution is bounded
  // above, below, both, or neither.
  void SSS::find_limits(double x) {
    logp_slice_ = logf_(x) - rexp_mt(rng(), 1.0);
    check_finite(x, logp_slice_);
    bool limits_successfully_found = true;
    if (doubly_bounded()) {
      lo_ = lower_bound_;
      logplo_ = logf_(lo_);
      hi_ = upper_bound_;
      logphi_ = logf_(hi_);
    } else if (lower_bounded()) {
      lo_ = lower_bound_;
      logplo_ = logf_(lo_);
      limits_successfully_found = find_upper_limit(x);
    } else if (upper_bounded()) {
      limits_successfully_found = find_lower_limit(x);
      hi_ = upper_bound_;
      logphi_ = logf_(hi_);
    } else {  // unbounded
      limits_successfully_found = find_limits_unbounded(x);
    }
    check_slice(x);
    if (limits_successfully_found) {
      check_probs(x);
    }
  }

  // Find the upper and lower limits of a slice containing x for a
  // potentially multimodal distribution.  Uses Neal's (2003 Annals of
  // Statistics) doubling algorithm.
  bool SSS::find_limits_unbounded(double x) {
    hi_ = x + suggested_dx_;
    lo_ = x - suggested_dx_;
    logphi_ = logf_(hi_);
    logplo_ = logf_(lo_);
    if (unimodal_) {
      find_limits_unbounded_unimodal(x);
      return true;
    } else {
      int doubling_count = 0;
      while (!done_doubling()) {
        double u = runif_mt(rng(), -1, 1);
        if (u > 0)
          double_hi(x);
        else
          double_lo(x);
        if (++doubling_count > 100) {
          // The slice has been doubled 100 times.  This is almost
          // certainly beecause of an error in the target distribution
          // or a crazy starting value.
          return false;
        }
      }
    }
    check_upper_limit(x);
    check_lower_limit(x);
    return true;
  }

  // A utility function used by find_limits_unbounded.
  bool SSS::done_doubling() const {
    return (logphi_ < logp_slice_) && (logplo_ < logp_slice_);
  }

  // Find the upper and lower limits of a slice when the target
  // distribution is known to be unimodal.
  void SSS::find_limits_unbounded_unimodal(double x) {
    hi_ = x + suggested_dx_;
    logphi_ = logf_(hi_);
    while (logphi_ >= logp_slice_) double_hi(x);
    check_upper_limit(x);

    lo_ = x - suggested_dx_;
    logplo_ = logf_(lo_);
    while (logplo_ >= logp_slice_) double_lo(x);
    check_lower_limit(x);
  }

  bool SSS::find_upper_limit(double x) {
    hi_ = x + suggested_dx_;
    logphi_ = logf_(hi_);
    int doubling_count = 0;
    while (logphi_ >= logp_slice_ || (!unimodal_ && runif_mt(rng()) > .5)) {
      double_hi(x);
      if (++doubling_count > 100) {
        // The slice has been doubled over 100 times.  This is almost
        // certainly because of an error in the implementation of the
        // target distribution, or a crazy starting value.
        return false;
      }
    }
    check_upper_limit(x);
    return true;
  }

  bool SSS::find_lower_limit(double x) {
    lo_ = x - suggested_dx_;
    logplo_ = logf_(lo_);
    int doubling_count = 0;
    while (logplo_ >= logp_slice_ || (!unimodal_ && runif_mt(rng()) > .5)) {
      double_lo(x);
      if (++doubling_count > 100) {
        // The slice has been doubled over 100 times.  This is almost
        // certainly because of an error in the implementation of the
        // target distribution, or a crazy starting value.
        return false;
      }
    }
    check_lower_limit(x);
    return true;
  }

  std::string SSS::error_message(double lo, double hi, double x, double logplo,
                                 double logphi, double logp_slice) const {
    ostringstream err;
    err << endl
        << "lo = " << lo << "  logp(lo) = " << logplo << endl
        << "hi = " << hi << "  logp(hi) = " << logphi << endl
        << "x  = " << x << "  logp(x)  = " << logp_slice << endl;
    return err.str().c_str();
  }

  void SSS::handle_error(const std::string &msg, double x) const {
    report_error(msg + " in ScalarSliceSampler" +
                 error_message(lo_, hi_, x, logplo_, logphi_, logp_slice_));
  }

  // Makes the upper end of the slice twice as far away from x, and
  // updates the density value
  void SSS::double_hi(double x) {
    double dx = hi_ - x;
    hi_ = x + 2 * dx;
    if (!std::isfinite(hi_)) {
      handle_error("infinite upper limit", x);
    }
    logphi_ = logf_(hi_);
  }

  // Makes the lower end of the slice twice as far away from x, and
  // updates the density value
  void SSS::double_lo(double x) {
    double dx = x - lo_;
    lo_ = x - 2 * dx;
    if (!std::isfinite(lo_)) handle_error("infinite lower limit", x);
    logplo_ = logf_(lo_);
  }

  //------ Quality assurance and error handling  ---------------------
  void SSS::check_slice(double x) {
    if (x < lo_ || x > hi_)
      handle_error("problem building slice:  x out of bounds", x);
    if (lo_ > hi_) handle_error("problem building slice:  lo > hi", x);
  }

  void SSS::check_probs(double x) {
    // logp may be infinite at the upper or lower bound
    bool logood = lower_bounded() || (logplo_ <= logp_slice_);
    bool higood = upper_bounded() || (logphi_ <= logp_slice_);
    if (logood && higood) return;
    handle_error("problem with probabilities", x);
  }

  bool SSS::lower_bounded() const { return lo_set_manually_; }
  bool SSS::upper_bounded() const { return hi_set_manually_; }
  bool SSS::doubly_bounded() const {
    return lo_set_manually_ && hi_set_manually_;
  }
  bool SSS::unbounded() const {
    return !(lo_set_manually_ || hi_set_manually_);
  }

  void SSS::check_finite(double x, double logp_slice_) {
    if (std::isfinite(logp_slice_)) return;
    handle_error("initial value leads to infinite probability", x);
  }

  void SSS::check_upper_limit(double x) {
    if (x > hi_) handle_error("x beyond upper limit", x);
    if (!std::isfinite(hi_)) handle_error("upper limit is infinite", x);
    if (isnan(logphi_)) handle_error("upper limit givs NaN probability", x);
  }

  void SSS::check_lower_limit(double x) {
    if (x < lo_) handle_error("x beyond lower limit", x);
    if (!std::isfinite(lo_)) handle_error("lower limit is infininte", x);
    if (isnan(logplo_)) handle_error("lower limit givs NaN probability", x);
  }

}  // namespace BOOM

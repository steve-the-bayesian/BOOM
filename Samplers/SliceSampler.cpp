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
#include "Samplers/SliceSampler.hpp"
#include <cassert>
#include <cmath>
#include <stdexcept>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"  // for mean()

namespace BOOM {
  SliceSampler::SliceSampler(const Func &log_density, bool Unimodal)
      : unimodal_(Unimodal), logp_(log_density) {
    hi_ = lo_ = scale_ = 1.0;
  }

  // To be called as part of draw().  Set up the bits that define the
  // slice, and make sure everything is finite.
  void SliceSampler::initialize() {
    // Check that logp_slice_ has a finite value.
    log_p_slice_ = logp_(last_position_);
    if (!std::isfinite(log_p_slice_)) {
      std::string msg = "invalid condition used to initialize SliceSampler";
      report_error(msg);
    }
    log_p_slice_ -= rexp_mt(rng(), 1);

    // Reset scale_ if something has gone wrong.
    if (scale_ < .0001 * fabs(mean(last_position_))) {
      // Very small values of scale_ can make the algorithm take
      // forever.
      scale_ = .1 * fabs(mean(last_position_));
    }
    if (scale_ <= 0.0 || !std::isfinite(scale_)) {
      // Infinite or NaN values of scale can result in an infinite
      // loop.
      scale_ = 1.0;
    }
    lo_ = scale_;
    hi_ = scale_;

    set_random_direction();
    logplo_ = logp_(last_position_ - lo_ * random_direction_);
    // If necessary, shrink the lower bound until the log density is
    // finite.
    while (!std::isfinite(logplo_)) {
      lo_ /= 2.0;
      logplo_ = logp_(last_position_ - lo_ * random_direction_);
    }

    logphi_ = logp_(last_position_ + hi_ * random_direction_);
    // If necessary, shrink the upper bound until the log density is
    // finite.
    while (!std::isfinite(logphi_)) {
      hi_ /= 2.0;
      logphi_ = logp_(last_position_ + hi_ * random_direction_);
    }
  }

  void SliceSampler::set_random_direction() {
    random_direction_.resize(last_position_.size());
    for (uint i = 0; i < random_direction_.size(); ++i) {
      random_direction_[i] = scale_ * rnorm();
    }
  }

  // Repeatedly choose one of lo_ or hi_ at random, and double it
  // until both lo_ and hi_ are out of the slice.
  void SliceSampler::doubling(bool upper) {
    int sgn = upper ? 1 : -1;
    double &value(upper ? hi_ : lo_);
    double old = value;
    double &p(upper ? logphi_ : logplo_);
    if (value <= 0.0) {
      report_error(
          "The slice sampler has collapsed.  Initial value "
          "may be on the boundary of the parameter space.");
    }
    // Double the value of the endpoint, unless doing so would produce
    // an infinity.
    value *= 2.0;
    if (!std::isfinite(value)) {
      value = old;
      return;
    }
    p = logp_(last_position_ + sgn * value * random_direction_);
    while (isnan(p)) {
      value = (value + old) / 2;
      p = logp_(last_position_ + sgn * value * random_direction_);
    }
  }

  void SliceSampler::find_limits() {
    if (unimodal_) {
      // If the posterior is unimodal then expand each endpoint until it
      // is out of the slice.
      while (logphi_ > log_p_slice_) doubling(true);
      while (logplo_ > log_p_slice_) doubling(false);
    } else {
      // If the posterior is not known to be unimodal, then randomly
      // pick an endpoint and double it until both ends are out of the
      // slice.  This will sometimes result in unnecessary doubling,
      // but this algorithm has the right stationary distribution
      // (Neal 2003).
      while (logphi_ > log_p_slice_ || logplo_ > log_p_slice_) {
        double tmp = runif_mt(rng(), -1, 1);
        doubling(tmp > 0);
      }
    }
  }

  // To be called when the candidate value is outside the slice.  If
  // the candidate is outside on the high end, then update the slice
  // so that the candidate defines the new upper end, and similarly if
  // the candidate is out on the low end.
  //
  // Note, this procedure is the complement of the "doubling"
  // procedure used to explore the slice boundaries.  Just as doubling
  // might miss some islands of probability mass in multi-model
  // distributions, "contract" might leave some behind.  However,
  // stochastic convergence of this algorithm is proved in Neal (2003).
  void SliceSampler::contract(double lambda, double logp_candidate) {
    if (lambda < 0) {
      lo_ = fabs(lambda);
      logplo_ = logp_candidate;
    } else if (lambda > 0) {
      hi_ = lambda;
      logphi_ = logp_candidate;
    }
  }

  Vector SliceSampler::draw(const Vector &theta) {
    last_position_ = theta;
    initialize();
    find_limits();
    Vector candidate;
    double logp_candidate = log_p_slice_ - 1;
    do {
      double lambda = runif_mt(rng(), -lo_, hi_);
      candidate = last_position_ + lambda * random_direction_;
      logp_candidate = logp_(candidate);
      if (logp_candidate < log_p_slice_) contract(lambda, logp_candidate);
    } while (logp_candidate < log_p_slice_);
    scale_ = hi_ + lo_;  // both hi_ and lo_ > 0
    return candidate;
  }

}  // namespace BOOM

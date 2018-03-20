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

#include "distributions/BoundedAdaptiveRejectionSampler.hpp"
#include <sstream>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace {
    typedef BoundedAdaptiveRejectionSampler BARS;

    // A streaming operator for vector<double>, mainly used for recording the
    // state of the BARS object in the event an exception is thrown.
    ostream &operator<<(ostream &out, const std::vector<double> &v) {
      for (int i = 0; i < v.size(); ++i) {
        out << v[i] << " ";
      }
      return out << std::endl;
    }
  }  // namespace

  BARS::BoundedAdaptiveRejectionSampler(
      double support_lower_bound,
      const std::function<double(double)> &log_target_density,
      const std::function<double(double)> &log_target_density_derivative)
      : log_target_density_(log_target_density),
        log_target_density_derivative_(log_target_density_derivative),
        x_(1, support_lower_bound),
        log_density_values_(1, log_target_density_(support_lower_bound)),
        log_density_derivative_values_(
            1, log_target_density_derivative_(support_lower_bound)),
        knots_(1, support_lower_bound) {
    if (log_density_derivative_values_[0] >= 0) {
      std::ostringstream err;
      err << "lower bound of " << support_lower_bound
          << " must be to the right of the mode of "
          << "logf in BoundedAdaptiveRejectionSampler" << std::endl
          << "a        = " << support_lower_bound << std::endl
          << "logf(a)  = " << log_density_values_[0] << std::endl
          << "dlogf(a) = " << log_density_derivative_values_[0] << std::endl;
      report_error(err.str());
    }
    update_cdf();
  }

  //----------------------------------------------------------------------
  void BARS::add_point(double z) {
    auto it = std::lower_bound(knots_.begin(), knots_.end(), z);

    if (it == knots_.end()) {
      x_.push_back(z);
      log_density_values_.push_back(log_target_density_(z));
      log_density_derivative_values_.push_back(
          log_target_density_derivative_(z));
    } else {
      uint k = it - knots_.begin();
      x_.insert(x_.begin() + k, z);
      log_density_values_.insert(log_density_values_.begin() + k,
                                 log_target_density_(z));
      log_density_derivative_values_.insert(
          log_density_derivative_values_.begin() + k,
          log_target_density_derivative_(z));
    }

    refresh_knots();
    update_cdf();
  }
  //----------------------------------------------------------------------
  void BARS::refresh_knots() {
    // wasteful!  should only update a knot between the x's, but
    // adding an x will change two knots
    knots_.resize(x_.size());
    knots_[0] = x_[0];
    for (uint i = 1; i < knots_.size(); ++i) {
      knots_[i] = compute_knot(i);
    }
  }
  //----------------------------------------------------------------------
  // returns the location of the intersection of the tanget line at
  // x_[k] and x_[k - 1]
  double BARS::compute_knot(uint k) const {
    double y2 = log_density_values_[k];
    double y1 = log_density_values_[k - 1];
    double d2 = log_density_derivative_values_[k];
    double d1 = log_density_derivative_values_[k - 1];
    double x2 = x_[k];
    double x1 = x_[k - 1];

    // If d2 == d1 then you've reached a spot of exponential decay, or
    // else x2 == x1.
    if (d2 == d1) return x1;

    double ans = (y1 - d1 * x1) - (y2 - d2 * x2);
    ans /= (d2 - d1);
    return ans;
  }
  //----------------------------------------------------------------------
  void BARS::update_cdf() {
    // cdf_[i] is the integral of the outer hull from knots_[i] to
    // knots_[i + 1], where the last value is implicitly infinity.

    // The cdf is un-normalized, so we divide everything by exp(y0).
    uint n = knots_.size();
    cdf_.resize(n);
    double y0 = log_density_values_[0];
    if (!std::isfinite(y0)) {
      report_error("log density value 0 is not finite.");
    }
    double last = 0;
    for (uint k = 0; k < knots_.size(); ++k) {
      double d = log_density_derivative_values_[k];
      double y = log_density_values_[k] - y0;
      double z = x_[k];
      double dinv = 1.0 / d;
      double inc1 = k == n - 1 ? 0 : dinv * exp(y - d * z + d * knots_[k + 1]);
      double inc2 = dinv * exp(y - d * z + d * knots_[k]);
      cdf_[k] = last + inc1 - inc2;
      last = cdf_[k];
      if (!std::isfinite(last)) {
        report_error(
            "BoundedAdaptiveRejectionSampler found an illegal value "
            "when updating the cdf.");
      }
    }
  }
  //----------------------------------------------------------------------
  double BARS::outer_hull(double z, uint k) const {
    double xk = x_[k];
    double dk = log_density_derivative_values_[k];
    double yk = log_density_values_[k];
    return yk + dk * (z - xk);
  }
  //----------------------------------------------------------------------
  double BARS::draw_safely(RNG &rng, int available_recursion_levels) {
    if (available_recursion_levels-- < 0) {
      ostringstream err;
      err << "Too many recursion layers in "
          << "BoundedAdaptiveRejectionSampler::draw" << std::endl;
      print(err);
      report_error(err.str());
      return negative_infinity();
    }
    double u = runif_mt(rng, 0, cdf_.back());
    auto pos = std::lower_bound(cdf_.begin(), cdf_.end(), u);
    uint k = pos - cdf_.begin();
    double cand;
    if (k + 1 == cdf_.size()) {
      // one sided draw..................
      cand = knots_.back() +
             rexp_mt(rng, -1 * log_density_derivative_values_.back());
    } else {
      // draw from the doubly truncated exponential distribution
      double lo = knots_[k];
      double hi = knots_[k + 1];
      double lam = -1 * log_density_derivative_values_[k];
      cand = rtrun_exp_mt(rng, lam, lo, hi);
    }
    double target = log_target_density_(cand);
    double hull = outer_hull(cand, k);
    double logu = hull - rexp_mt(rng, 1);
    // The <= in the following statement is important in edge cases
    // where you're very close to the boundary.
    if (logu <= target) return cand;
    add_point(cand);
    return draw_safely(rng, available_recursion_levels);
  }

  double BARS::draw(RNG &rng) { return draw_safely(rng, 1000); }

  std::ostream &BARS::print(std::ostream &out) const {
    out << "proposed points: " << endl
        << x_ << endl
        << "log density " << endl
        << log_density_values_ << endl
        << "knots = " << endl
        << knots_ << endl
        << "cdf = " << endl
        << cdf_ << endl;
    return (out);
  }
}  // namespace BOOM

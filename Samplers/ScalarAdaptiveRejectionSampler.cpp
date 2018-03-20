// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Samplers/ScalarAdaptiveRejectionSampler.hpp"
#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  ScalarAdaptiveRejectionSampler::ScalarAdaptiveRejectionSampler(
      const std::function<double(double)> &logf)
      : log_density_(logf) {}

  void ScalarAdaptiveRejectionSampler::add_point(double x) {
    if (x < log_density_approximation_.lower_limit() ||
        x > log_density_approximation_.upper_limit()) {
      report_error("Illegal point added to density approximation.");
    }
    log_density_approximation_.add_point(x, log_density_(x));
  }

  double ScalarAdaptiveRejectionSampler::draw() {
    ensure_approximation_is_initialized();
    double candidate = log_density_approximation_.sample(rng());

    // Sample a uniform random number between 0 and the height of the
    // envelope, on the log scale.
    double log_u =
        log_density_approximation_.envelope(candidate) + log(runif_mt(rng()));

    // If the uniform number is less than the lower bound then accept.
    // Otherwise, accept if it is less than the density function.
    // Checking vs. the lower bound is a computational savings,
    // because it is almost certainly cheaper to computer than the
    // density function.
    if (log_u < log_density_approximation_.lower_bound(candidate) ||
        log_u < log_density_(candidate)) {
      return candidate;
    } else {
      add_point(candidate);
      return this->draw();
    }
  }

  double ScalarAdaptiveRejectionSampler::logp(double x) const {
    return log_density_(x);
  }

  void ScalarAdaptiveRejectionSampler::ensure_approximation_is_initialized() {
    double lo = log_density_approximation_.lower_limit();
    double hi = log_density_approximation_.upper_limit();
    bool lo_is_infinite = lo == BOOM::negative_infinity();
    bool hi_is_infinite = hi == BOOM::infinity();

    while (log_density_approximation_.number_of_knots() < 3) {
      // Generate a random candidate point.
      double point = 0;
      if (lo_is_infinite) {
        if (hi_is_infinite) {
          // Both lo and hi are infinite.
          point = rcauchy_mt(rng(), 0, 1);
        } else {
          // Domain is (-infinity, hi].
          const Vector &knots(log_density_approximation_.knots());
          point = (knots.empty() ? hi : knots[0]) - rexp_mt(rng(), 1.0);
        }
      } else {
        // Here lo is finite.
        if (hi_is_infinite) {
          // Domain is [lo, infinity).
          const Vector &knots(log_density_approximation_.knots());
          point = (knots.empty() ? lo : knots.back()) + rexp_mt(rng(), 1.0);
        } else {
          // Both lo and hi are finite.
          point = runif_mt(rng(), lo, hi);
        }
      }
      add_point(point);
    }  // closes while loop

    if (lo_is_infinite) {
      // If the lower limit is infinite, we need two points to the
      // left of the mode, so we can ensure that the left tail
      // probability of the approximation is finite.  We get there by
      // adding points on the left hand side at progressively smaller
      // values.
      int counter = 0;
      while (log_density_approximation_.log_density_values_at_knots()[1] <
             log_density_approximation_.log_density_values_at_knots()[0]) {
        const Vector &knots(log_density_approximation_.knots());
        double dx = std::max<double>(1.0, knots[1] - knots[0]);
        add_point(knots[0] - 2 * dx);
        if (++counter > 50) {
          report_error("Too many doubling attempts on left side.");
        }
      }
    }

    if (hi_is_infinite) {
      // As above, but for the right side.  I.e. we need to make sure
      // that we've got at least two points to the right of the mode
      // so we can ensure a negative slope and thus a finite right
      // tail probability.
      int counter = 0;
      while (log_density_approximation_.log_density_values_at_knots()
                 [log_density_approximation_.number_of_knots() - 2] <
             log_density_approximation_.log_density_values_at_knots()
                 [log_density_approximation_.number_of_knots() - 1]) {
        const Vector &knots(log_density_approximation_.knots());
        int nk = log_density_approximation_.number_of_knots();
        double dx = std::max<double>(1.0, knots.back() - knots[nk - 2]);
        add_point(knots.back() + 2 * dx);
        if (++counter > 50) {
          report_error("Too many doubling attempts on riht side.");
        }
      }
    }
  }

  namespace ARS {
    typedef PiecewiseExponentialApproximation PEA;

    PEA::PiecewiseExponentialApproximation()
        : lower_limit_(negative_infinity()), upper_limit_(infinity()) {}

    void PEA::set_lower_limit(double lo) {
      if (lo > upper_limit_) {
        report_error("Lower limit cannot exceed upper limit.");
      }
      lower_limit_ = lo;
    }
    void PEA::set_upper_limit(double hi) {
      if (hi < lower_limit_) {
        report_error("Upper limit cannot be less than lower limit.");
      }
      upper_limit_ = hi;
    }

    void PEA::set_limits(double lo, double hi) {
      lower_limit_ = lo;
      upper_limit_ = hi;
      if (lo > hi) {
        report_warning("Adaptive rejection sampler had to swap limits.");
        std::swap(lower_limit_, upper_limit_);
      }
    }

    void PEA::add_point(double x, double logf) {
      if (!finite(x) || std::isnan(logf)) {
        report_error("Adding an illegal point.");
      }
      std::vector<double>::iterator b = knots_.begin();
      std::vector<double>::iterator knots_iterator =
          std::lower_bound(b, knots_.end(), x);
      int position_of_new_knot = 0;
      if (knots_iterator == knots_.end()) {
        position_of_new_knot = knots_.empty() ? 0 : knots_.size() - 1;
        knots_.push_back(x);
        logf_.push_back(logf);
      } else {
        if (x == *knots_iterator) {
          // If the point has already been added to the ensemble, no
          // need to add it again.
          return;
        }
        position_of_new_knot = knots_iterator - b;
        std::vector<double>::iterator logf_iterator =
            logf_.begin() + position_of_new_knot;
        knots_.insert(knots_iterator, x);
        logf_.insert(logf_iterator, logf);
      }
      update_region_probabilities(position_of_new_knot);
    }

    // Args:
    //   position_of_new_knot: Index of the new knot location, in the
    //     "old" vector of knots.
    void PEA::update_region_probabilities(int position_of_new_knot) {
      if (knots_.size() < 3) {
        // Until we have three knots, this object is uninitialized and
        // region probabilities cannot be computed.
      } else if (knots_.size() == 3) {
        // Initialize everything.
        log_region_probability_.resize(4);
        recompute_region_probabilities();
      } else {
        log_region_probability_.insert(
            log_region_probability_.begin() + position_of_new_knot, 0);
        recompute_region_probabilities();
      }
    }

    void PEA::recompute_region_probabilities() {
      for (int i = 0; i < log_region_probability_.size(); ++i) {
        log_region_probability_[i] =
            log_probability_between_adjacent_knots(i - 1);
      }
    }

    double PEA::lower_bound(double x) const {
      if (knots_.empty() || x < knots_[0] || x > knots_.back()) {
        return BOOM::negative_infinity();
      }
      // lower_bound(v, x) points to the position in v where you would
      // insert x.  Thus it is the first point greater than or equal
      // to x.  I.e. it is the right_knot in the interval containing
      // x.
      std::vector<double>::const_iterator it =
          std::lower_bound(knots_.begin(), knots_.end(), x);
      int right_knot = (it - knots_.begin());
      int left_knot = right_knot - 1;
      return interpolate(x, left_knot, right_knot);
    }

    double PEA::envelope(double x) const {
      const double inf = BOOM::infinity();
      int nk = number_of_knots();
      if (nk < 3) {
        report_error("Not enough knots to compute the envelope.");
      }
      if (x < knots_[0]) {
        return interpolate(x, 0, 1);
      } else if (x > knots_.back()) {
        return interpolate(x, nk - 2, nk - 1);
      } else {
        std::vector<double>::const_iterator b = knots_.begin();
        std::vector<double>::const_iterator it =
            std::lower_bound(b, knots_.end(), x);
        int right_knot = it - b;
        int left_knot = right_knot - 1;
        double left_line =
            (left_knot >= 1) ? interpolate(x, left_knot - 1, left_knot) : inf;
        double right_line = (right_knot + 1 < nk)
                                ? interpolate(x, right_knot, right_knot + 1)
                                : inf;
        return std::min(left_line, right_line);
      }
    }

    double PEA::interpolate(double x, int knot0, int knot1) const {
      double x0 = knots_[knot0];
      double y0 = logf_[knot0];
      double x1 = knots_[knot1];
      double y1 = logf_[knot1];
      if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
      }
      double slope = (y1 - y0) / (x1 - x0);
      return y0 + slope * (x - x0);
    }

    void PEA::interpolating_equation(int knot0, int knot1, double *intercept,
                                     double *slope) const {
      double x0 = knots_[knot0];
      double y0 = logf_[knot0];
      double x1 = knots_[knot1];
      double y1 = logf_[knot1];
      if (x0 > x1) {
        std::swap(x0, x1);
        std::swap(y0, y1);
      }
      *slope = (y1 - y0) / (x1 - x0);
      *intercept = y0 - x0 * (*slope);
    }

    // Return the log probability between lower_limit and upper_limit of
    // the un-normalized distribuiton exp(intercept + slope * x).
    double PEA::cumulative_exponential_log_probability(
        double intercept, double slope, double lower_limit,
        double upper_limit) const {
      if (lower_limit > upper_limit) {
        std::swap(lower_limit, upper_limit);
      }
      double ans = intercept;
      if (slope < 0) {
        ans += lde2(slope * lower_limit, slope * upper_limit) - log(-slope);
      } else if (slope > 0) {
        ans += lde2(slope * upper_limit, slope * lower_limit) - log(slope);
      } else {
        ans += log(upper_limit - lower_limit);
      }
      return ans;
    }

    // Returns the value of x where the interpolating lines between
    // left_knot and right_knot intersect, where right_knot is
    // left_knot + 1.
    double PEA::point_of_intersection(int left_knot) const {
      if (left_knot >= number_of_knots() - 1) {
        std::ostringstream err;
        err << "There is no knot after " << left_knot << "." << std::endl;
        report_error(err.str());
        return 0;
      } else if (left_knot < 0) {
        report_error("point_of_intersection called with a negative argument.");
        return 0;
      } else if (left_knot == 0) {
        // Between knots 0 and 1, there is a vertical line from
        // knot[0] to the secant line determined by knots 1 and 2.
        // These intersect at knot[0].
        return knots_[0];
      } else if (left_knot == number_of_knots() - 2) {
        // Between the last two knots, there is a vertical line
        // extending from the final knot intersecting the secant line
        // determined by the two preceding knots.  They intersect at
        // the final knot.
        return knots_.back();
      } else {
        double intercept0, slope0, intercept1, slope1;
        interpolating_equation(left_knot - 1, left_knot, &intercept0, &slope0);
        interpolating_equation(left_knot + 1, left_knot + 2, &intercept1,
                               &slope1);
        return (intercept1 - intercept0) / (slope0 - slope1);
      }
    }

    double PEA::log_probability_between_adjacent_knots(int left_knot) const {
      double slope, intercept;
      if (left_knot >= number_of_knots() || left_knot < -1) {
        report_error("knot out of bounds.");
        return 0.0;  // silence the compiler
      } else if (left_knot == -1) {
        // Initial, potentially unbounded region on left side.  The
        // interpolant is determined by knots 0 and 1.
        interpolating_equation(0, 1, &intercept, &slope);
        return cumulative_exponential_log_probability(intercept, slope,
                                                      lower_limit_, knots_[0]);
      } else if (left_knot == 0) {
        // First interior region on left side.  The interpolant is
        // determined by knots 1 and 2.
        interpolating_equation(1, 2, &intercept, &slope);
        return cumulative_exponential_log_probability(intercept, slope,
                                                      knots_[0], knots_[1]);
      } else if (left_knot == number_of_knots() - 2) {
        // Final interior region, on right side.  Interpolant is
        // determined by knots nk-3 and nk-2.
        interpolating_equation(left_knot - 1, left_knot, &intercept, &slope);
        return cumulative_exponential_log_probability(
            intercept, slope, knots_[left_knot], knots_[left_knot + 1]);
      } else if (left_knot == number_of_knots() - 1) {
        // Final, poentially unbounded region on right hand side.
        // Interpolant determined by nk-2 and nk-1.
        interpolating_equation(left_knot - 1, left_knot, &intercept, &slope);
        return cumulative_exponential_log_probability(
            intercept, slope, knots_[left_knot], upper_limit_);
      } else {
        // Buffered interior region.  The interpolant has two parts.
        // The first is determined by the left knot, and the knot to
        // its left.  The second is determined by the right knot, and
        // the knot to its right.
        double xstar = point_of_intersection(left_knot);
        interpolating_equation(left_knot - 1, left_knot, &intercept, &slope);
        double logp0 = cumulative_exponential_log_probability(
            intercept, slope, knots_[left_knot], xstar);

        int right_knot = left_knot + 1;
        interpolating_equation(right_knot, right_knot + 1, &intercept, &slope);
        double logp1 = cumulative_exponential_log_probability(
            intercept, slope, xstar, knots_[right_knot]);
        return lse2(logp0, logp1);
      }
    }

    double PEA::sample(RNG &rng) const {
      int which_region = randomly_choose_region(rng);
      double ans = sample_from_region(which_region, rng);
      if (!finite(ans) || std::isnan(ans)) {
        report_error("Bad simulation from piecewise linear approximation.");
      }
      return ans;
    }

    int PEA::randomly_choose_region(RNG &rng) const {
      Vector region_probabilities = log_region_probability_;
      region_probabilities.normalize_logprob();
      return rmulti_mt(rng, region_probabilities);
    }

    double PEA::sample_from_region(int which_region, RNG &rng) const {
      double intercept, slope, lo, hi;
      if (which_region < 0 || which_region > number_of_knots()) {
        report_error("region out of bounds.");
        return 0.0;  // silence the compiler
      } else if (which_region == 0) {
        // Draw from the truncated exponential between lower_limit_
        // and knots_[0].
        interpolating_equation(0, 1, &intercept, &slope);
        lo = lower_limit_;
        hi = knots_[0];
      } else if (which_region == 1) {
        // Draw from the truncated exponential between knots_[0] and knots_[1].
        interpolating_equation(1, 2, &intercept, &slope);
        lo = knots_[0];
        hi = knots_[1];
      } else if (which_region == number_of_knots() - 1) {
        // Draw from the truncated exponential between
        // number_of_knots() - 2 and number_of_knots() - 1.
        interpolating_equation(number_of_knots() - 3, number_of_knots() - 2,
                               &intercept, &slope);
        lo = knots_[number_of_knots() - 2];
        hi = knots_[number_of_knots() - 1];
      } else if (which_region == number_of_knots()) {
        // Draw from the truncated exponential between knots_.back()
        // and upper_limit_.
        interpolating_equation(number_of_knots() - 2, number_of_knots() - 1,
                               &intercept, &slope);
        lo = knots_.back();
        hi = upper_limit_;
      } else {
        // Between 2 interior knots.  Find the point of intersection.
        int left_knot = which_region - 1;
        int right_knot = which_region;
        double xstar = point_of_intersection(left_knot);
        double left_intercept, left_slope;
        interpolating_equation(left_knot - 1, left_knot, &left_intercept,
                               &left_slope);
        double logp_left = cumulative_exponential_log_probability(
            left_intercept, left_slope, knots_[left_knot], xstar);

        double right_intercept, right_slope;
        interpolating_equation(right_knot, right_knot + 1, &right_intercept,
                               &right_slope);
        double logp_right = cumulative_exponential_log_probability(
            right_intercept, right_slope, xstar, knots_[right_knot]);

        double log_normalizing_constant = lse2(logp_left, logp_right);
        double p_left = exp(logp_left - log_normalizing_constant);
        double u = runif_mt(rng);
        if (u < p_left) {
          intercept = left_intercept;
          slope = left_slope;
          lo = knots_[left_knot];
          hi = xstar;
        } else {
          intercept = right_intercept;
          slope = right_slope;
          lo = xstar;
          hi = knots_[right_knot];
        }
      }
      if (lo == BOOM::negative_infinity() && slope <= 0) {
        report_error("Density is increasing unboundedly to the left.");
      } else if (hi == BOOM::infinity() && slope >= 0) {
        report_error("Density is increasing unboundedly to the right.");
      }
      double ans = rpiecewise_log_linear_mt(rng, slope, lo, hi);
      if (std::isnan(ans)) {
        report_error(
            "Adaptive rejection sampler approximation "
            "simulated a NaN.");
      }
      return ans;
    }

  }  // namespace ARS
}  // namespace BOOM

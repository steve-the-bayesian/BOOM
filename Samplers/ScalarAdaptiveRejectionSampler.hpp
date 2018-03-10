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

#ifndef BOOM_SAMPLERS_SCALAR_ADAPTIVE_REJECTION_SAMPLER_HPP_
#define BOOM_SAMPLERS_SCALAR_ADAPTIVE_REJECTION_SAMPLER_HPP_

#include <functional>
#include "LinAlg/Vector.hpp"
#include "Samplers/Sampler.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  namespace ARS {

    // A piecewise linear approximation to the the log of a
    // log-concave density.  Lower and upper limits can be specified
    // for the domain of the distribution (though they default to
    // -infinity and infinity).
    class PiecewiseExponentialApproximation {
     public:
      PiecewiseExponentialApproximation();

      // Set lower and upper limits on the support of the approximation.
      // lower_limit must be less than the current upper_limit.
      void set_lower_limit(double lower_support_limit);

      // upper_support_limit must be greater than the current lower_limit.
      void set_upper_limit(double upper_support_limit);

      // Must have upper_limit > lower_limit.
      void set_limits(double lower_support_limit, double upper_support_limit);

      double lower_limit() const { return lower_limit_; }
      double upper_limit() const { return upper_limit_; }

      // Add the point (x, logf(x)) to the approximation.  It is an
      // error to call this function with x outside the range
      // (lower_limit, upper_limit).
      void add_point(double x, double log_f_of_x);

      // Return the lower bound for logf.  The lower bound is the
      // linear interpolation between the first knot less than x and
      // the first knot greater than x.  If x is less than the first
      // knot or greater than the last knot then the lower bound is
      // -infinity.
      double lower_bound(double x) const;

      // Returns the envelope approximation (the picewise linear upper
      // bound) to logf.  The envelope approximation is as follows:
      // 1) If x is less than the first knot, then the envelope
      //    approximation is the interpolation from the first two
      //    knots.  If lower_limit is -infinity then the first two
      //    knots must have an upward slope, or an exception is
      //    thrown.  It is the caller's responsibility to ensure an
      //    upward slope.
      // 2) If x is greater than the final knot, then the envelope
      //    approximation is the linear interpolation from the final
      //    two knots.  If upper_limit is infinity then the final two
      //    knots must have a downward slope, or an exception will be
      //    thrown.  It is the caller's responsibility to ensure a
      //    downward slope in this instance.
      // 3) If x is between the first and second knot then the envelope
      //    is the interpolation from the second and third knot.
      // 4) If x is between the next-to-last and last knot then the
      //    envelope is the interpolation from the third_to_last and
      //    next_to_last knots.
      // 5) Otherwise, the envelope is the smaller of the
      //    interpolations determined by the two points to the left of
      //    x and the two points to the right of x.
      //
      // Between the two extreme knots, the envelope looks like Bart
      // Simpson's hair.
      double envelope(double x) const;

      double sample(RNG &rng) const;

      int number_of_knots() const { return knots_.size(); }

      const Vector &knots() const { return knots_; }

      const Vector &log_density_values_at_knots() const { return logf_; }

     private:
      // The domain of the approximation consists of 5 kinds of
      // subsets.  There is an exponential tail to the left of the
      // first knot, and the right of the last knot.  The log density
      // is linear in each of the two "buffer" regions formed by the
      // first two (and last two) knots.  In all other ("non-buffer")
      // regions the log density is trapezoidal.

      // Returns an integer between 0 and number_of_knots(),
      // inclusive.  The selected region is the region to the left of
      // the returned knot.  So 0 corresponds to (-infinity,
      // knots_[0]), for example.  Region n is (knots_.back(),
      // infinity).
      int randomly_choose_region(RNG &rng) const;

      // Draws a random value from the specified region.
      // Args:
      //   which_region: Regions are numbered according the knot on
      //     their right.  The region between lower_limit and knots[0]
      //     is region 0.  The region between knots.back() and
      //     upper_limit is region 'number_of_knots()'.  It is an
      //     error to call this function if which_region is outside
      //     [0, number_of_knots()].
      //   rng:  A U(0, 1) random number generator.
      double sample_from_region(int which_region, RNG &rng) const;

      // Returns the log of the unnormalized probability mass between
      // knots[left_knot] and knots[left_knot + 1].
      double log_probability_between_adjacent_knots(int left_knot) const;

      // Return the log probability between lower_limit and upper_limit of
      // the un-normalized distribuiton exp(intercept + slope * x).
      // That is, this function returns the log of
      // integral(from = lower_limit,
      //          to = upper_limit,
      //          fun = exp(intercept + slope * x),
      //          integration_variable = x)
      double cumulative_exponential_log_probability(double intercept,
                                                    double slope,
                                                    double lower_limit,
                                                    double upper_limit) const;

      // Returns the value of the linear function defined by the two
      // specified knots.  In most cases knot2 = knot1 + 1.  Note that
      // x will typically be a value in the region immediately to the
      // left or to the right of the interval defined by (knot1,
      // knot2).
      double interpolate(double x, int knot1, int knot2) const;

      // Computes the slope and intercept of the line connecting the
      // two specified knots.
      // Args:
      //   knot0:  Index of the left hand knot.  0 <= knot0 < knot1.
      //   knot1:  Index of the right hand knot.
      //     knot0 < knot1 < number_of_knots()
      //   intercept:  On output, the intercept of the interpolating line.
      //   slope:  On output, the slope of the interpolating line.
      void interpolating_equation(int knot0, int knot1, double *intercept,
                                  double *slope) const;

      // Returns the value of x where the interpolating lines between
      // left_knot and right_knot intersect, where right_knot is
      // left_knot + 1.
      // Args:
      //   left_knot: The left knot of the interval containing the
      //     intersection.  The intersection will occur at a knot
      //     location in either of the buffer regions, and it will
      //     occur inside the region at any interior region that is
      //     not a buffer region.  It is an error to call this
      //     function with left_knot < 0 or left_knot >=
      //     number_of_knots().
      double point_of_intersection(int left_knot) const;

      // When a new knot is added at x, find the region containing x,
      // split its log probability into two pieces, and update
      // log_region_probability_.
      void update_region_probabilities(int position_of_new_knot);

      // Recompute the probabilities for all regions.
      void recompute_region_probabilities();

      // The knots are the sorted "x" values of the approximation points.
      Vector knots_;

      // logf_ contains the "y" values of the approximation points.
      // It is the same length as knots_, and logf_[j] ==
      // logf(knots_[j]).
      Vector logf_;

      // This vector has one more element than does knots_ and logf_.
      // Each entry is proportional to the log of probability mass
      // contained in the given region number, where regions are
      // numbered by the knot on thier right (e.g. region 0 ends with
      // knots_[0], region 1 ends with knots_[1], etc).  Region
      // number_of_knots() begins with knots_.back(), and ends with
      // upper_limit.
      Vector log_region_probability_;

      // Lower and upper limits on the distribution's region of support.
      double lower_limit_;
      double upper_limit_;
    };

  }  // namespace ARS

  class ScalarAdaptiveRejectionSampler : public ScalarSampler {
   public:
    explicit ScalarAdaptiveRejectionSampler(
        const std::function<double(double)> &log_density);

    void set_lower_limit(double lo) {
      log_density_approximation_.set_lower_limit(lo);
    }
    void set_upper_limit(double hi) {
      log_density_approximation_.set_upper_limit(hi);
    }
    void set_limits(double lo, double hi) {
      log_density_approximation_.set_limits(lo, hi);
    }

    // Adds the point x to the approximate density.
    void add_point(double x);

    double draw();
    double draw(double) override { return draw(); }  // ignore argument

    virtual double logp(double x) const;

   private:
    // If the approximation has fewer than 3 knots, add knots.
    // * If lower_limit_ and upper_limit_ are finite, then the first 3
    //   are selected uniformly at random from the range of support.
    // * If both limits are infinite then the first 3 knots are drawn
    //   from the Cauchy distribution.
    // * If one limit is finite, the first three knots are sampled
    //   from that limit plus (or minus) a gamma random variable.
    // If there are three or more knots, then keep adding new knots at
    // greater and greater increments until there is a decreasing
    // slope in the direction of infinite support.
    void ensure_approximation_is_initialized();

    std::function<double(double)> log_density_;
    ARS::PiecewiseExponentialApproximation log_density_approximation_;
  };

}  // namespace BOOM

#endif  // BOOM_SAMPLERS_SCALAR_ADAPTIVE_REJECTION_SAMPLER_HPP_

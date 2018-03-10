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
#ifndef BOOM_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP
#define BOOM_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP

#include <functional>
#include <vector>
#include "distributions.hpp"

namespace BOOM {

  // For sampling from a log-concave, strictly decreasing function.  Implements
  // the adaptive rejection sampling algorithm from Gilks et. al. 1993 for
  // functions which are known to be log concave.  Computational support is
  // limited to functions with (mathematical) support above a truncation point
  // on the real line.  For the implementation used here, the target function
  // must be strictly decreasing beyond the truncation point.
  class BoundedAdaptiveRejectionSampler {
   public:
    // Args:
    //   support_lower_bound: The left endpoint of the support for the target
    //     distribution.  Density values to the left of support_lower_bound are
    //     assumed to be zero.  support_lower_bound must be to the right of
    //     the mode of log_target_density.
    //   log_target_density: The log of the (potentially unnormalized) density
    //     function to be sampled.  log_target_density must be a concave
    //     function( e.g -x^2).
    //   log_target_density_derivative:  The derivative of log_target_density.
    BoundedAdaptiveRejectionSampler(
        double support_lower_bound,
        const std::function<double(double)> &log_target_density,
        const std::function<double(double)> &log_target_density_derivative);

    // Simulate a draw from the target distribution using adaptive rejection
    // sampling.  If the initial proposal is not accepted, the approximating
    // function is modified by adding the proposal to the approximation set
    // (improving the approximation), and then another proposal is made.  The
    // process is repeated until a successful proposal is made.
    double draw(RNG &);

    std::ostream &print(std::ostream &out) const;

   private:
    std::function<double(double)> log_target_density_;
    std::function<double(double)> log_target_density_derivative_;

    // The points that have been tried thus far, stored in ascending order.
    std::vector<double> x_;

    // Function values corresponding to values in x_.
    std::vector<double> log_density_values_;

    // Derivatives of the log target density evaluated at x_.
    std::vector<double> log_density_derivative_values_;

    // Contains the points of intersection between the lines that are tangent to
    // log_target_density at x_.  First knot is x_[0].  Later knots satisfy
    // x_[i-1] < knots_[i] < x_[i].
    std::vector<double> knots_;

    // Grid containting the cumulative density funciton of the outer hull
    // approximation.  cdf_[i] = cdf_[i-1] + the integral of the hull from
    // knots[i] to knots_[i+1].  cdf.back() assumes a final knot at infinity.
    std::vector<double> cdf_;

    // The implementation of the draw() method, which is recursive.  The
    // available_recursion_levels argument starts off at a big number, which
    // gets decremented at each recursion level.  If the recursion gets too deep
    // an exception is thrown.
    double draw_safely(RNG &rng, int available_recursion_levels);

    // Add the point x to the approximation set, improving the approximation.
    void add_point(double x);

    // Evaluate the outer hull approximation at x.
    double outer_hull(double x, uint k) const;

    // To be called after a point is added to the approximation.  Refreshes all
    // the values in the cdf_ array.
    void update_cdf();

    // Recomputes the location of all the elements in knots_.
    void refresh_knots();

    // Returns the location (i.e. the x coordinate) of the intersection of the
    // lines tangent to the log target density at x_[k] and x_[k - 1].
    double compute_knot(uint k) const;
  };

}  // namespace BOOM
#endif  // BOOM_BOUNDED_ADAPTIVE_REJECTION_SAMPLER_HPP

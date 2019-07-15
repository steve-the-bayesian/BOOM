// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#ifndef BOOM_MSPLINE_HPP_
#define BOOM_MSPLINE_HPP_

#include "stats/Spline.hpp"

namespace BOOM {

  // A spline class that is closely related to B-splines, sharing many of the
  // same locality properties.  Each Mspline basis function is nonzero over
  // 'order' knot ranges.  Each spline basis function is integrates to 1 and is
  // thus the pdf for a random variable with local support.
  //
  // Msplines are reviewed in Ramsay (1988) Statistical Science pp425--461.
  class Mspline : public SplineBase {
   public:
    // Args:
    //   knots: The vector of knots defining the mesh points for the spline.
    //     The smallest and largest values determine the interval over which the
    //     spline is defined.  These knots are replicated an infinite number of
    //     times.  In particular, knots 0...order-1 are copies of the smallest
    //     element in the vector.  This differs from the scheme used in Bspline,
    //     but matches the notation used in Ramsay (1988).  This implementation
    //     of Msplines does not allow duplicated interior knots, so the elements
    //     of the 'knots' vector should be distinct.  They will be sorted by the
    //     base class constructor.
    //   order: The number of coefficients required to evaluate the M-polynomial
    //     in x.  For a cubic spline the order is 4.
    explicit Mspline(const Vector &knots, int order = 4);

    // Knots are counted starting from the order() replicated knots at the left
    // endpoint of the spline interval.
    double knot(int i) const override {
      return SplineBase::knot(i - order_ + 1);
    }

    int number_of_interior_knots() const {
      return std::max<int>(SplineBase::number_of_knots() - 2, 0);
    }

    // Counting scheme for 'notional knots' at the end of the interval matches
    // Ramsay (1988).
    int number_of_knots() const override {
      return number_of_interior_knots() + 2 * order();
    }

    int knot_span(double x) const override {
      return SplineBase::knot_span(x) + order() - 1;
    }

    int order() const { return order_; }

    // Return an Mspline basis function expansion at the value x.
    Vector basis(double x) const override;

    // The dimension of the vector returned by basis(x).
    int basis_dimension() const override { return basis_dimension_; }

    // Evaluates M_k(x), where k can be 0, 1, ... number_of_knots + order - 1.
    // Args:
    //   x: The argument where the knot basis element will be evaluated.
    //   order:  The order of the spline function to be evaluated.
    //   which_basis_element: The index of the specific basis function to
    //     evaluate.
    double mspline_basis_function(double x, int order,
                                  int which_basis_element) const;

   private:
    void increment_basis_dimension() override;
    void decrement_basis_dimension() override;
    int order_;
    int basis_dimension_;
  };

  // An Ispline is an integrated Mspline.  Each basis function is the CDF of a
  // random variable with local support, and is thus a monotonic function.  If
  // the basis functions are combined linearly with all positive coefficients
  // then the result is a strictly increasing function.
  //
  // Note the inheritance here.  An Ispline does not have an is-a relationship
  // with an Mspline, so public inheritance is probably not the right
  // relationship, but it does mean that the knots, etc living in the SplineBase
  // base class of the Mspline get used by the Ispline.
  class Ispline : public Mspline {
   public:
    // The 'order' of an Ispline is the order of the underlying Mspline.
    // Args:
    //   knots: The knots defining the spline basis.  The first and last knots
    //     are the left and right endpoints of the interval over which the
    //     spline is defined.  They are assumed to repeat infinitely many times.
    //     Interior knots must be unique (that is a requirement of this
    //     implementation, not a mathematical requirement).
    //   order:  The order of the underlying Mspline.
    explicit Ispline(const Vector &knots, int order = 4);

    // Returns the Ispline basis expansion of the value x.
    Vector basis(double x) const override;

    // Evaluate a particular Ispline basis function.
    // Args:
    //   x: The location (function argument) where the spline basis function
    //     should be evaluated.
    //   order:  The order of the Ispline basis function to evaluate.
    //   which_basis_element: The index of the basis function to evaluate.
    // Returns:
    //   The value of the indicated basis function at x.
    double ispline_basis_function(double x, int order,
                                  int which_basis_element) const;
  };

}  // namespace BOOM

#endif  //  BOOM_MSPLINE_HPP_

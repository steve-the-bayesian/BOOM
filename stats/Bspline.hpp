// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2015 Steven L. Scott

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

#ifndef BOOM_STATS_BSPLINE_HPP_
#define BOOM_STATS_BSPLINE_HPP_

#include "LinAlg/Vector.hpp"
#include "stats/Spline.hpp"

namespace BOOM {

  // Compute a Bspline basis expansion of a scalar value x.  A Bspline is a
  // spline formed by a set of local basis functions (the b-spline basis)
  // determined by a set of knots.  The knots partition an interval [lo, hi]
  // over which the spline function nonzero.  The spline basis is zero over
  // (-infinity, lo) and (hi, infinity).
  //
  // To make the B-spline theory work, the notional set of knots is supposed to
  // be infinite, but in practice a B-spline basis function of degree d is
  // nonzero over at most d+1 knot spans.  This means we can add d+1 fake knots
  // at the beginning and the end of the knot sequence, and the knots can be in
  // any arbitrary positions.  This class follows an established convention of
  // adding the fake knots at the first and last elements of the knot vector.
  class Bspline : public SplineBase {
   public:
    // Args:
    //   knots: The set of knots for the spline.  In between pairs of knots, the
    //     spline is a piecewise polynomial whose degree is given by the second
    //     argument.  The first and last knots define the interval over which
    //     the spline is defined.  These knots are implicitly replicated an
    //     infinite number of times.
    //   degree: The degree of the piecewise polynomial in between pairs of
    //     interior knots.
    explicit Bspline(const Vector &knots, int degree = 3);

    // The Bspline basis expansion at the value x.  If x lies outside the range
    // [knots.begin(), knots.end()] then all basis elements are zero.
    //
    // If there are fewer than 2 knots then the return value is empty.
    Vector basis(double x) const override;

    // The dimension of the spline basis, which is one for every distinct
    // interval covered by knots(), plus one for every degree of the piecewise
    // polynomial.  Normally this is number_of_knots - 1 + degree, though it can
    // be less if knots() contains duplicate elements.  If knots().size <= 1
    // then the basis_dimension is 0.
    int basis_dimension() const override { return basis_dimension_; }

    // The order of the piecewise polynomial connecting the knots.
    int order() const { return order_; }

    // The degree of the piecewise polynomial connecting the knots.
    int degree() const { return order_ - 1; }

    // Compute the coefficient C for combining two splines of order degree-1
    // into a spline of order degree.  The recursion is
    // basis[i, degree](x) =
    //     C(x, i, degree) * basis[i, degree-1]
    // + (1 - C(x, i + 1, degree)) * basis[i + 1, degree - 1].
    // See deBoor, page 90, chapter IX, formulas (15) and (16).
    double compute_coefficient(double x, int knot_span, int degree) const;

   private:
    // The order (1 + degree) of the piecewise polynomial connecting the knots.
    int order_;

    // The dimension of the spline basis expansion.
    int basis_dimension_;

    void increment_basis_dimension() override { ++basis_dimension_; }
    void decrement_basis_dimension() override { --basis_dimension_; }
  };

}  // namespace BOOM

#endif  //  BOOM_STATS_BSPLINE_HPP_

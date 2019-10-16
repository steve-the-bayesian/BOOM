// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2016 Steven L. Scott

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
#ifndef BOOM_SPLINE_HPP
#define BOOM_SPLINE_HPP
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"

namespace BOOM {
  // A base class providing features shared by different spline bases.
  // Examples inclue Bsplines, Msplines, and Isplines.
  class SplineBase {
   public:
    // Args:
    //   knots: The set of knots for the spline.  Between pairs of
    //     knots, the spline is a piecewise polynomial whose degree is
    //     given by the second argument.
    //
    // Splines also have a notion of 'degree' or 'order', but this can
    // be defined differently by different bases, so it is not part of
    // the shared base class.
    explicit SplineBase(const Vector &knots);
    virtual ~SplineBase() {}

    // The spline basis expansion (value of each basis function) at x.
    virtual Vector basis(double x) const = 0;

    // Row i of the returned matrix is the basis expansion of x[i].
    Matrix basis_matrix(const Vector &x) const;

    // The dimension of the spline basis (i.e. the dimension of the
    // vector returned by a call to 'basis()'.
    virtual int basis_dimension() const = 0;

    // Adds a knot at the given  Location.  If knot_location lies
    // before the first or after the last current knot, then the
    // domain of the spline is extended to cover knot_location.
    void add_knot(double knot_location);

    // Remove the specified knot.  An exception will be thrown if
    // which_knot is outside the range of knots_.  If which_knot == 0
    // or which_knot == number_of_knots() - 1 then the domain of the
    // spline basis will be reduced.
    void remove_knot(int which_knot);

    // The vector of knots.  Implicit boundary knots are not included.
    virtual const Vector &knots() const { return knots_; }
    virtual int number_of_knots() const { return knots_.size(); }

    // If the argument is in the interior of the knots vector, return
    // knots_[i].  If it is off the end to the left return knots_[0].
    // If it is off the end to the right then include knots_.back().
    // The implicit assumption is that we have an infinite set of
    // knots piled up on the beginning and end of the actual knot
    // sequence.
    virtual double knot(int i) const;

    virtual double final_knot() const;

    // Compute the index of the largest knot less than or equal to x.
    virtual int knot_span(double x) const;

   private:
    virtual void increment_basis_dimension() = 0;
    virtual void decrement_basis_dimension() = 0;

    // The vector of knots defining the mesh points for the spline.
    // This is sorted by the constructor and it is kept sorted by
    // add_knot() and remove_knot().  The terminal knots at the
    // beginning and end of this vector are assumed to repeat an
    // infinite number of times.  If a knot is added outside the knot
    // range, then the old terminal knot becomes a single knot, and
    // the new terminal knot is infinitely repeated.
    Vector knots_;
  };

}  // namespace BOOM
#endif  // BOOM_SPLINE_HPP

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
#ifndef BOOM_NATURAL_SPLINE_HPP
#define BOOM_NATURAL_SPLINE_HPP
#include "LinAlg/Matrix.hpp"
#include "LinAlg/QR.hpp"
#include "LinAlg/Vector.hpp"

#include <vector>

namespace BOOM {
  class NaturalSpline {
    // cubic "natural" spline
   public:
    NaturalSpline(double lo, double hi, uint nInterKnots, bool icpt = false);
    NaturalSpline(const Vector &Knots, double lo, double hi, bool icpt = false);
    double predict(double x, const Vector &beta) const;
    Vector operator()(double x) const;  // returns natural spline basis
    Vector knots() const;
    int nknots() const;
    uint basis_dim() const;

   private:
    static const int order_ = 4;
    static const int ordm1_ = 3;

    Vector remove_intercept(const Vector &b) const;

    mutable int curs;       // current position in knots vector
    mutable bool boundary;  // must have knots[curs] <= x < knots[curs+1]
                            // except for the boundary case
    Vector knots_;          // knots
    mutable Vector rdel;
    mutable Vector ldel;
    mutable Vector a;  // scratch array
    mutable int offsets;
    mutable Vector wsp;
    const bool icpt;

    QR qr_const;  // for removing the constant term
    Vector basis_left;
    Vector deriv_left;
    Vector basis_right;
    Vector deriv_right;

    Vector basis_interior(double x, uint nder = 0) const;
    Vector basis_exterior(double x, uint nder = 0) const;
    Vector basis(double x, uint nder = 0) const;
    const Vector &minimal_basis(double x, uint nder = 0) const;
    // returns the set of nonzero functions, a vector of length order_

    double eval_derivs(double x, int nder) const;  // can have nder=0;
    void set_cursor(double x) const;
    void basis_funcs(double x, Vector &ans) const;
    void diff_table(double x, int ordm1) const;
    bool in_outer_knots(double x) const;
    QR make_qr_const(double lo, double hi) const;

    void too_few_knots() const;
  };

  // make natural spline basis matrix
  Matrix ns(const Vector &x, uint df);  // equally spaced knots.
  Matrix ns(const Vector &x, const Vector &InteriorKnots, double lo, double hi);

}  // namespace BOOM
#endif  // BOOM_NATURAL_SPLINE_HPP

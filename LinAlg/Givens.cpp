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

#include <cmath>
#include <iomanip>
#include <iostream>

#include "LinAlg/EigenMap.hpp"
#include "LinAlg/Givens.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"

#include "Eigen/Jacobi"

namespace BOOM {
  using std::endl;
  using std::setw;
  typedef GivensRotation GR;

  // A version of drot, hand translated from drot.f on netlib.
  void drot(int n, double *x, int x_stride, double *y, int y_stride, double c,
            double s) {
    if (n <= 0) return;
    if (x_stride == 1 && y_stride == 1) {
      for (int i = 0; i < n; ++i) {
        double dtmp = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = dtmp;
      }
    } else {
      // Handle the case where at least one of the strides is not 1.
      int ix = 0;
      int iy = 0;
      for (int i = 0; i < n; ++i) {
        double dtmp = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = dtmp;
        ix += x_stride;
        iy += y_stride;
      }
    }
  }

  void givens(double a, double b, double &c, double &s);
  void givens(double a, double b, double &c, double &s) {
    if (b == 0) {
      c = 1;
      s = 0;
      return;
    }
    double ab = std::fabs(b);
    double aa = std::fabs(a);
    if (ab > aa) {
      double r = -a / b;
      s = 1.0 / std::sqrt(1 + r * r);
      c = s * r;
      return;
    } else {
      double r = -b / a;
      c = 1 / std::sqrt(1 + r * r);
      s = c * r;
    }
  }

  GR::GivensRotation(int I, int J, double C, double S)
      : i(I), j(J), c(C), s(S) {}

  GR::GivensRotation(const Matrix &A, int I, int J) : i(I), j(J), c(0), s(0) {
    double a(A(j, j));
    double b(A(i, j));
    givens(a, b, c, s);
  }

  GR GR::trans() const { return GR(i, j, c, -s); }

  Matrix &operator*(const GR &G, Matrix &A) {
    // transpose G before multiplying?
    // pre-multiplication affects the rows of A
    VectorView x(A.row(G.i));
    VectorView y(A.row(G.j));
    // EigenMap(A.row(G.i)).applyOnTheLeft(Eigen::JacobiRotation<double>(G.c,
    // G.s));
    // EigenMap(A.row(G.j)).applyOnTheLeft(Eigen::JacobiRotation<double>(G.c,
    // G.s));
    drot(x.size(), x.data(), x.stride(), y.data(), y.stride(), G.c, G.s);
    return A;
  }

  Matrix &operator*(Matrix &A, const GR &G) {
    // EigenMap(A.col(G.i)).applyOnTheRight(Eigen::JacobiRotation<double>(G.c,
    // -(G.s)));
    // EigenMap(A.col(G.j)).applyOnTheRight(Eigen::JacobiRotation<double>(G.c,
    // -(G.s)));
    VectorView x(A.col(G.i));
    VectorView y(A.col(G.j));
    drot(x.size(), x.data(), x.stride(), y.data(), y.stride(), G.c, -(G.s));
    return A;
  }

  ostream &GR::print(ostream &out) const {
    out << setw(10) << c << " " << setw(10) << s << endl
        << setw(10) << -s << " " << setw(10) << c << endl;
    return out;
  }

  ostream &operator<<(ostream &out, const GR &G) { return G.print(out); }

  Matrix triangulate(const Matrix &U, const Selector &inc,
                     bool chop_zero_rows) {
    // U is an upper triangular matrix.  inc selects columns of U.
    // Then we apply Givens rotations to the subdiagonal elements to
    // produce a rectangular upper triangular matrix.
    Matrix R = inc.select_cols(U);
    uint nvars = inc.nvars();
    for (uint c = 0; c < nvars; ++c) {
      uint nr = inc.indx(c);
      for (uint r = nr; r > c; --r) {
        GivensRotation G(R, r, c);
        R = G * R;
      }
    }
    if (!chop_zero_rows) return R;

    Matrix ans(nvars, nvars);
    for (uint i = 0; i < nvars; ++i) ans.row(i) = R.row(i);
    return ans;
  }
}  // namespace BOOM

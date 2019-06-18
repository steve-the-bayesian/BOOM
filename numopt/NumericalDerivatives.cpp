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
#include "numopt/NumericalDerivatives.hpp"
#include <cmath>
#include <limits>
#include "LinAlg/SpdMatrix.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  NumericalDerivatives::NumericalDerivatives(const Target &f) : f_(f) {}

  // The value for h was taken from:
  // http://journal.info.unlp.edu.ar/journal/journal6/papers/ipaper.pdf
  Vector NumericalDerivatives::gradient(const Vector &x) const {
    int dim = x.size();
    Vector g(dim);
    const double tol = cbrt(std::numeric_limits<double>::epsilon());
    for (int i = 0; i < dim; ++i) {
      double h = tol * std::max<double>(0.1, fabs(x[i]));
      g[i] = scalar_first_derivative(x, i, h);
    }
    return g;
  }

  // A Richardson approximation to the first derivative.  For
  // derivation, see
  // http://www2.math.umd.edu/~dlevy/classes/amsc466/lecture-notes/differentiation-chap.pdf
  double NumericalDerivatives::scalar_first_derivative(const Vector &x, int pos,
                                                       double h) const {
    Vector dx(x);
    dx[pos] = x[pos] + h;
    double fp1 = f_(dx);
    dx[pos] = x[pos] - h;
    double fm1 = f_(dx);
    dx[pos] = x[pos] + 2 * h;
    double fp2 = f_(dx);
    dx[pos] = x[pos] - 2 * h;
    double fm2 = f_(dx);

    double df = -fp2 + 8 * fp1 - 8 * fm1 + fm2;
    return df / (12 * h);
  }

  Matrix NumericalDerivatives::Hessian(const Vector &x,
                                       bool quick_and_dirty) const {
    int dim = x.size();
    SpdMatrix ans(x.size());
    const double tol = cbrt(std::numeric_limits<double>::epsilon());
    for (int i = 0; i < dim; ++i) {
      double hi = tol * std::max<double>(0.1, fabs(x[i]));
      int lo = quick_and_dirty ? i : 0;
      for (int j = lo; j < dim; ++j) {
        double hj = tol * std::max<double>(0.1, fabs(x[j]));
        if (i == j) {
          ans(i, j) = homogeneous_scalar_second_derivative(x, i, hi);
        } else {
          ans(i, j) = scalar_second_derivative(x, i, hi, j, hj);
        }
      }
    }
    if (quick_and_dirty) {
      ans.reflect();
    } else {
      ans = .5 * (ans + ans.transpose());
    }
    return std::move(ans);
  }

  double NumericalDerivatives::homogeneous_scalar_second_derivative(
      const Vector &x, int pos, double h) const {
    Vector dx(x);
    double f0 = f_(x);
    dx[pos] = x[pos] + h;
    double fp = f_(dx);
    dx[pos] = x[pos] - h;
    double fm = f_(dx);
    return (fp + fm - 2 * f0) / square(h);
  }

  // Using the central second derivative found here:
  // http://terminus.sdsu.edu/SDSU/Math693a_f2005/Lectures/16/lecture-static-04.pdf
  double NumericalDerivatives::scalar_second_derivative(const Vector &x, int i,
                                                        double hi, int j,
                                                        double hj) const {
    if (i == j) {
      report_error("Call homogeneous_scalar_second_derivative instead.");
    }
    Vector dx(x);
    dx[i] = x[i] + hi;
    dx[j] = x[j] + hj;
    // pp, pm, etc indicate whether x[i] and x[j] are plus or minus.
    double f_plus_plus = f_(dx);
    dx[j] = x[j] - hj;
    double f_plus_minus = f_(dx);
    dx[i] = x[i] - hi;
    double f_minus_minus = f_(dx);
    dx[j] = x[j] + hj;
    double f_minus_plus = f_(dx);
    return (+f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) /
           (4 * hi * hj);
  }

  ScalarNumericalDerivatives::ScalarNumericalDerivatives(const ScalarTarget &f)
      : f_(f) {}

  double ScalarNumericalDerivatives::first_derivative(double x) const {
    const double tol = cbrt(std::numeric_limits<double>::epsilon());
    double h = tol * std::max<double>(0.1, fabs(x));
    double fp1 = f_(x + h);
    double fm1 = f_(x - h);
    double fp2 = f_(x + 2 * h);
    double fm2 = f_(x - 2 * h);
    double df = -fp2 + 8 * fp1 - 8 * fm1 + fm2;
    return df / (12 * h);
  }

  double ScalarNumericalDerivatives::second_derivative(double x) const {
    const double tol = cbrt(std::numeric_limits<double>::epsilon());
    double h = tol * std::max<double>(0.1, fabs(x));
    double f0 = f_(x);
    double fp = f_(x + h);
    double fm = f_(x - h);
    return (fp + fm - 2 * f0) / square(h);
  }

  NumericJacobian::NumericJacobian(const Mapping &inverse_transformation)
      : inverse_transformation_(inverse_transformation) {}

  namespace {
    class SubFunction {
     public:
      typedef NumericJacobian::Mapping Mapping;
      SubFunction(const Mapping &mapping, int position)
          : mapping_(mapping), position_(position) {}

      double operator()(const Vector &x) { return mapping_(x)[position_]; }

     private:
      Mapping mapping_;
      int position_;
    };
  }  // namespace

  Matrix NumericJacobian::matrix(const Vector &z) {
    int dim = z.size();
    Matrix ans(dim, dim);
    for (int i = 0; i < dim; ++i) {
      SubFunction f(inverse_transformation_, i);
      NumericalDerivatives derivatives(f);
      ans.col(i) = derivatives.gradient(z);
    }
    return ans;
  }

}  // namespace BOOM

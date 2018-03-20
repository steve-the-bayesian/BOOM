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

#ifndef BOOM_NUMERICAL_DERIVATIVES_HPP_
#define BOOM_NUMERICAL_DERIVATIVES_HPP_

#include <functional>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {

  class NumericalDerivatives {
   public:
    typedef std::function<double(const Vector &)> Target;
    explicit NumericalDerivatives(const Target &f);

    // Returns the gradient of f at the point x.
    Vector gradient(const Vector &x) const;

    // Hessian matrix (matrix of second partial derivatives) of f at
    // x.  Mathematically the Hessian matrix is symmetric.  If
    // quick_and_dirty is true then this function will only compute
    // the upper triangle of the hessian, and then reflect it into the
    // lower triangle.  It can be more precise (but is more expensive)
    // to compute both triangles and average them, which is what is
    // done if quick_and_dirty is false.
    Matrix Hessian(const Vector &x, bool quick_and_dirty = false) const;

   private:
    // First partial derivative of f with respect to x[pos].
    // Args:
    //   x: The location where the derivative is to be taken.
    //   pos:  Ordinate of x with which to differentiate.
    //   h:  Step size to use in the approximation.
    double scalar_first_derivative(const Vector &x, int pos, double h) const;

    // Second partial derivative of f with respect to x[i] and x[j].
    // Separate step sizes are used.
    double scalar_second_derivative(const Vector &x, int i, double hi, int j,
                                    double hj) const;

    // Second partial derivative of f with respect to x[i].
    double homogeneous_scalar_second_derivative(const Vector &x, int pos,
                                                double h) const;

    Target f_;
  };

  // Compute the first and second derivatives of a scalar target function.
  class ScalarNumericalDerivatives {
   public:
    typedef std::function<double(double)> ScalarTarget;
    explicit ScalarNumericalDerivatives(const ScalarTarget &f);
    double first_derivative(double x) const;
    double second_derivative(double x) const;

   private:
    ScalarTarget f_;
  };

  // Compute the Jacobian of a mapping.  This is mainly intended for
  // testing to make sure the math is right for analytic Jacobian
  // objects.
  class NumericJacobian {
   public:
    typedef std::function<Vector(const Vector &)> Mapping;
    explicit NumericJacobian(const Mapping &inverse_transformation);

    // Returns the derivative of each element of
    // inverse_transformation(new_parameterization) with respect to
    // each element of new_parameterization.  The matrix is organized
    // with elements of new_parameterization on the rows, and elements
    // of inverse_transformation(new_parameterization) on the columns.
    //
    // Note: This is the transpose of the way Jacobians are often
    // defined, but using this method, gradient(fz(z)) = Jacobian *
    // gradient(fx(x)), where z is the new parameterization and x is
    // the old.
    Matrix matrix(const Vector &new_parameterization);

   private:
    Mapping inverse_transformation_;
  };

}  // namespace BOOM

#endif  //  BOOM_NUMERICAL_DERIVATIVES_HPP_

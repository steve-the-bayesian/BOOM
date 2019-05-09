#ifndef BOOM_TEST_UTILITIES_CHECK_DERIVATIVES_HPP_
#define BOOM_TEST_UTILITIES_CHECK_DERIVATIVES_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include <functional>
#include <string>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {
  using DerivativeTestTarget =
      std::function<double(const Vector &,  Vector &, Matrix &, int)>;
  // Check the math for analytically computed gradients and Hessians.
  //
  // If numeric derivatives match the analytic deriviatves at the evaluation
  // point then an empty sring is returned.  Otherwise an error message is
  // returned.  
  //
  // The intent is for this function to be used as 
  // EXPECT_EQ("", CheckDerivatives(...))
  //
  // Args:
  //   target: A twice-differentiable function whose derivatives should be
  //     checked for correctness.  The first argument is the location where the
  //     derivatives should be evaluated.  The second is the gradient, and the
  //     third is the Hessian.  The final argument is the number of desired
  //     derivatives (0, 1, or 2).
  //   evaluation_point: The point where the function and its derivatives should
  //     be evaluated.
  //   epsilon: The maximum absolute distance that the numeric derivatives can
  //     be from the analytic derivatives at any coordinate.  This should be a
  //     small number, but not too small because the numerical derivatives will
  //     exhibit some amount of numerical instability.
  std::string CheckDerivatives(DerivativeTestTarget target,
                               const Vector &evaluation_point,
                               double epsilon = 1e-3);

  // As above, but with an alternate framework for specifying the
  // twice-differentiable function.
  using PointerDerivativeTestTarget =
      std::function<double(const Vector &, Vector *, Matrix *)>;
  std::string CheckDerivatives(PointerDerivativeTestTarget target,
                               const Vector &evaluation_point,
                               double epsilon = 1e-3);


  // Check the derivatives of a scalar valued function.
  using ScalarDerivativeTestTarget =
      std::function<double(double, double &, double &, int)>;
  std::string CheckDerivatives(ScalarDerivativeTestTarget target,
                               double evaluation_point,
                               double epsilon = 1e-3);
  
}  // namespace BOOM

#endif  // BOOM_TEST_UTILITIES_CHECK_DERIVATIVES_HPP_

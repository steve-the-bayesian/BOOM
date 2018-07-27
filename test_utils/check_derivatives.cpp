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

#include "test_utils/check_derivatives.hpp"
#include "test_utils/test_utils.hpp"
#include "numopt/NumericalDerivatives.hpp"

namespace BOOM {

  std::string CheckDerivatives(DerivativeTestTarget target,
                               const Vector &evaluation_point,
                               double epsilon) {
    using std::endl;
    NumericalDerivatives derivs( [&target](const Vector &x) {
        Vector g;
        Matrix h;
        return target(x, g, h, 0);
      });

    int dim = evaluation_point.size();
    Vector analytic_gradient(dim);
    Matrix analytic_hessian(dim, dim);
    target(evaluation_point, analytic_gradient, analytic_hessian, 2);
    std::ostringstream err;
    Vector numeric_gradient = derivs.gradient(evaluation_point);
    if (!VectorEquals(analytic_gradient, numeric_gradient, epsilon)) {
      err << "gradient does not match." << endl
          << "analytic    numeric" << endl
          << cbind(analytic_gradient, numeric_gradient);
      return err.str();
    }

    Matrix numeric_hessian = derivs.Hessian(evaluation_point);
    if (!MatrixEquals(analytic_hessian, numeric_hessian, epsilon)) {
      err << "Hessian does not match." << endl
          << "Analytic Hessian: " << endl
          << analytic_hessian
          << "Numeric Hessian: " << endl
          << numeric_hessian;
      return err.str();
    }
    return "";
  }

  std::string CheckDerivatives(PointerDerivativeTestTarget &target,
                               const Vector &evaluation_point,
                               double epsilon) {
    using std::endl;
    DerivativeTestTarget real_target =
        [&target](const Vector &v, Vector &gradient, Matrix &hessian, uint nd) {
      return target(v, nd > 0 ? &gradient : nullptr, nd > 1 ? &hessian : nullptr);
    };
    return CheckDerivatives(real_target, evaluation_point, epsilon);
  }
  
}  // namespace BOOM



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
#include <sstream>

namespace BOOM {

  namespace {

    // If everything in g and h is within epsilon of 0 return true.  Otherwise
    // return false.
    bool all_zeros(const Vector &g, const Matrix &h, double epsilon) {
      for (int i = 0; i < g.size(); ++i) {
        if (fabs(g[i]) > epsilon) return false;
      }

      for (int i = 0; i < h.nrow(); ++i) {
        for (int j = 0; j < h.ncol(); ++j) {
          if (fabs(h(i, j)) > epsilon) {
            return false;
          }
        }
      }
      return true;
    }
  }  // namespace

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
    if (all_zeros(analytic_gradient, analytic_hessian, epsilon)) {
      return "Test function was constant at evaluation point.";
    }

    Vector numeric_gradient = derivs.gradient(evaluation_point);
    if (!VectorEquals(analytic_gradient, numeric_gradient, epsilon)) {
      err << "gradient does not match." << endl
          << "analytic    numeric" << endl
          << cbind(analytic_gradient, numeric_gradient)
          << "maximum absolute deviation: "
          << (numeric_gradient - analytic_gradient).max_abs();
      return err.str();
    }

    Matrix numeric_hessian = derivs.Hessian(evaluation_point);
    if (!MatrixEquals(analytic_hessian, numeric_hessian, epsilon)) {
      err << "Hessian does not match." << endl
          << "Analytic Hessian: " << endl
          << analytic_hessian
          << "Numeric Hessian: " << endl
          << numeric_hessian
          << "maximum absolute deviation: "
          << (numeric_hessian - analytic_hessian).max_abs();

      return err.str();
    }
    return "";
  }

  //===========================================================================
  std::string CheckDerivatives(PointerDerivativeTestTarget &target,
                               const Vector &evaluation_point,
                               double epsilon) {
    using std::endl;
    DerivativeTestTarget real_target =
        [&target](const Vector &v, Vector &gradient, Matrix &hessian, int nd) {
      return target(v, nd > 0 ? &gradient : nullptr, nd > 1 ? &hessian : nullptr);
    };
    return CheckDerivatives(real_target, evaluation_point, epsilon);
  }

  //===========================================================================
  std::string CheckDerivatives(ScalarDerivativeTestTarget target,
                               double evaluation_point,
                               double epsilon) {

    // Compute analytic derivatives.
    double g = 0, h = 0;
    target(evaluation_point, g, h, 2);

    if (fabs(g) < epsilon && fabs(h) < epsilon) {
      return "Test function was constant at the evaluation point.";
    }

    ScalarNumericalDerivatives derivs([target](double x) {
        double g, h;
        return target(x, g, h, 0);
      });

    // Check first derivative.
    double d1_numeric = derivs.first_derivative(evaluation_point);
    std::ostringstream err;
    if (fabs(g - d1_numeric) > epsilon) {
      err << "first derivative does not match." << std::endl
          << "analytic:    " << g << std::endl
          << "numeric:     " << d1_numeric << std::endl;
      return err.str();
    }

    // Check second derivative.
    double d2_numeric = derivs.second_derivative(evaluation_point);
    if (fabs(h - d2_numeric) > epsilon) {
      err << "second derivative does not match." << std::endl
          << "analytic:   " << h << std::endl
          << "numeric:    " << d2_numeric << std::endl;
      return err.str();
    }

    return "";
  }

}  // namespace BOOM

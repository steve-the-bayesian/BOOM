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

#ifndef BOOM_JACOBIAN_CHECKER_HPP_
#define BOOM_JACOBIAN_CHECKER_HPP_

#include <functional>
#include <memory>
#include "TargetFun/Transformation.hpp"
#include "numopt/NumericalDerivatives.hpp"

namespace BOOM {

  // Uses numerical derivatives to verify that an analytic Jacobian
  // class got the math right.
  class JacobianChecker {
   public:
    // Args:
    //   inverse_transformation: A mapping from the new
    //     parameterization back to the original parameterization.
    //     Example: you've got a distribution parameterized (mu,
    //     sigma^2) and you want to model it as (mu, log sigma^2).
    //     Then inverse_transformation({arg1, arg2}) returns {arg1,
    //     exp(arg2)}.
    //   analytic_jacobian: The Jacobian modeling the derivatives of
    //     inverse_transformation.
    //   epsilon: Used to define the neighborhood around two objects
    //     when deciding whether the numerical version and the
    //     analytic version match.
    JacobianChecker(
        const std::function<Vector(const Vector &)> &transformation,
        const std::function<Vector(const Vector &)> &inverse_transformation,
        const std::shared_ptr<Jacobian> &analytic_jacobian,
        double epsilon);

    void set_epsilon(double eps) {
      epsilon_ = eps;
    }

    // The following functions return 'true' if the analytic and
    // numeric Jacobians match, and 'false' if they don't.

    // Checks that the numeric Jacobian matrix is within epsilon of
    // the analytic Jacobian matrix across all elements.
    bool check_matrix(const Vector &new_parameterization);

    // Checks that the analytic logdet() is within epsilon of the log
    // determinant of the numeric Jacobian matrix.
    bool check_logdet(const Vector &new_parameterization);

    // Checks whether the analytic and numeric second derivatives of
    // the Jacobian matrix match within epsilon.
    //
    // Returns the empty string if second_order_element matches its numerical
    // derivatives sufficiently closely.  Otherwise an error message is returned
    // detailing the curcumstances of the failed check.
    std::string check_second_order_elements(const Vector &new_parameterization);

    // Checks that the analytic Jacobian matrix correctly transforms
    // the second order gradient.
    // Args:
    //   new_parameterization: The point at which to evaluate the
    //     transform_second_order_gradient function.
    //   error_message: If not NULL then a pointer to a string where
    //     an error message can be written.
    bool check_transform_second_order_gradient(
        const Vector &new_parameterization,
        std::string *error_message = nullptr);

    // Checks that the analytic Jacobian matrix computes the gradient
    // and Hessian of logdet() correctly.  These checks are based on
    // numeric derivatives of analytic_jacobian->logdet(), so be sure
    // to check that first.
    bool check_logdet_gradient(const Vector &new_parameterization);
    bool check_logdet_Hessian(const Vector &new_parameterization);

   private:
    std::function<Vector(const Vector &)> transformation_;
    std::function<Vector(const Vector &)> inverse_transformation_;
    NumericJacobian numeric_jacobian_;
    std::shared_ptr<Jacobian> analytic_jacobian_;
    double epsilon_;
  };

}  // namespace BOOM

#endif  //  BOOM_JACOBIAN_CHECKER_HPP_

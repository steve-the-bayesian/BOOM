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

#include "TargetFun/JacobianChecker.hpp"
#include <sstream>
#include <string>

namespace {
  typedef std::function<BOOM::Vector(const BOOM::Vector &)> Mapping;
}

namespace BOOM {
  JacobianChecker::JacobianChecker(
      const std::function<Vector(const Vector &)> &transformation,
      const std::function<Vector(const Vector &)> &inverse_transformation,
      const std::shared_ptr<Jacobian> &analytic_jacobian, double epsilon)
      : transformation_(transformation),
        inverse_transformation_(inverse_transformation),
        numeric_jacobian_(inverse_transformation_),
        analytic_jacobian_(analytic_jacobian),
        epsilon_(epsilon) {}

  bool JacobianChecker::check_matrix(const Vector &new_parameterization) {
    Vector original_parameterization = inverse_transformation_(
        new_parameterization);

    Matrix numeric_matrix = numeric_jacobian_.matrix(new_parameterization);
    Matrix analytic_matrix = analytic_jacobian_->matrix(original_parameterization);
    Matrix difference = analytic_matrix - numeric_matrix;
    return difference.max_abs() < epsilon_;
  }

  bool JacobianChecker::check_logdet(const Vector &new_parameterization) {
    Vector original_params = inverse_transformation_(new_parameterization);
    Matrix jake = analytic_jacobian_->matrix(original_params);
    return fabs(log(fabs(det(jake))) -
                analytic_jacobian_->logdet(original_params)) < epsilon_;
  }

  namespace {
    class SubFunction {
     public:
      SubFunction(const Mapping &inverse_mapping, int position)
          : inverse_mapping_(inverse_mapping), position_(position) {}

      double operator()(const Vector &new_parameterization) {
        return inverse_mapping_(new_parameterization)[position_];
      }

     private:
      Mapping inverse_mapping_;
      int position_;
    };
  }  // namespace

  std::string JacobianChecker::check_second_order_elements(
      const Vector &new_parameterization) {
    int dim = new_parameterization.size();
    std::vector<Matrix> numeric_hessians;
    Vector original_params = inverse_transformation_(new_parameterization);
    for (int t = 0; t < dim; ++t) {
      SubFunction ft(inverse_transformation_, t);
      NumericalDerivatives derivatives(ft);
      // The entry in numeric hessians[t](r, s) is the second derivative of the
      // old parameterization with respect to elements r and s of the new
      // parameterization.
      //
      // If math was done correctly, this element should match
      // analytic_jacobian_->second_order_element(r, s, t, original_params).
      numeric_hessians.push_back(derivatives.Hessian(new_parameterization));
    }

    std::string error_message = "";
    for (int r = 0; r < dim; ++r) {
      for (int s = 0; s < dim; ++s) {
        for (int t = 0; t < dim; ++t) {
          double analytic_second_derivative =
              analytic_jacobian_->second_order_element(r, s, t, original_params);
          if (fabs(analytic_second_derivative - numeric_hessians[t](r, s)) >
              epsilon_) {
            std::ostringstream err;
            err << "Element (" << r << "," << s << "," << t << ")"
                << " had a numeric second derivative of "
                << numeric_hessians[t](r, s)
                << " but an analytic second derivative of "
                << analytic_second_derivative << "." << std::endl;
            error_message += err.str();
          }
        }
      }
    }
    return error_message;
  }

  // The default implementation of transform_second_order_gradient is
  // correct, but slow.  Check that the method used in the concrete
  // class matches the answer you get from the reliable, slow method
  // from the base class.  Note, the reliable, slow method depends on
  // the correctness of analytic_jacobian_->second_order_element().
  bool JacobianChecker::check_transform_second_order_gradient(
      const Vector &new_parameterization, std::string *error_message) {
    int dim = new_parameterization.size();
    Vector original_params = inverse_transformation_(new_parameterization);
    Vector original_gradient(dim);
    original_gradient.randomize();
    SpdMatrix working_hessian(dim);
    working_hessian.randomize();

    Vector original_gradient_copy(original_gradient);
    SpdMatrix working_hessian_copy(working_hessian);

    analytic_jacobian_->transform_second_order_gradient(
        working_hessian, original_gradient, original_params);
    analytic_jacobian_->Jacobian::transform_second_order_gradient(
        working_hessian_copy, original_gradient_copy, original_params);

    if ((working_hessian_copy - working_hessian).max_abs() > epsilon_) {
      if (error_message) {
        std::ostringstream err;
        err << "Answer from concrete class:" << std::endl
            << working_hessian << std::endl
            << "Answer from base class:" << std::endl
            << working_hessian_copy << std::endl;
        *error_message = err.str();
      }
      return false;
    }
    return true;
  }

  namespace {
    class LogDet {
     public:
      explicit LogDet(const std::shared_ptr<Jacobian> &analytic_jacobian,
                      const Mapping &inverse_transformation)
          : analytic_jacobian_(analytic_jacobian),
            inverse_transformation_(inverse_transformation)
      {}

      double operator()(const Vector &new_parameterization) {
        Vector original_params = inverse_transformation_(new_parameterization);
        return analytic_jacobian_->logdet(original_params);
      }

     private:
      std::shared_ptr<Jacobian> analytic_jacobian_;
      Mapping inverse_transformation_;
    };
  }  // namespace

  bool JacobianChecker::check_logdet_gradient(
      const Vector &new_parameterization) {
    Vector original_params = inverse_transformation_(new_parameterization);
    Vector analytic_gradient = new_parameterization * 0;
    analytic_jacobian_->add_logdet_gradient(analytic_gradient, original_params);

    LogDet analytic_logdet(analytic_jacobian_, inverse_transformation_);
    NumericalDerivatives derivatives(analytic_logdet);
    Vector numeric_gradient = derivatives.gradient(new_parameterization);
    return (numeric_gradient - analytic_gradient).max_abs() < epsilon_;
  }

  bool JacobianChecker::check_logdet_Hessian(
      const Vector &new_parameterization) {
    Vector original_params = inverse_transformation_(new_parameterization);
    int dim = new_parameterization.size();
    Matrix analytic_hessian(dim, dim, 0.0);
    analytic_jacobian_->add_logdet_Hessian(analytic_hessian, original_params);

    LogDet analytic_logdet(analytic_jacobian_, inverse_transformation_);
    NumericalDerivatives derivatives(analytic_logdet);
    Matrix numeric_hessian = derivatives.Hessian(new_parameterization);
    return (numeric_hessian - analytic_hessian).max_abs() < epsilon_;
  }

}  // namespace BOOM

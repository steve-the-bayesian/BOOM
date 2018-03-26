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

namespace BOOM {
  JacobianChecker::JacobianChecker(
      const std::function<Vector(const Vector &)> &inverse_transformation,
      const std::shared_ptr<Jacobian> &analytic_jacobian, double epsilon)
      : inverse_transformation_(inverse_transformation),
        numeric_jacobian_(inverse_transformation_),
        analytic_jacobian_(analytic_jacobian),
        epsilon_(epsilon) {}

  bool JacobianChecker::check_matrix(const Vector &new_parameterization) {
    Matrix numeric_matrix = numeric_jacobian_.matrix(new_parameterization);
    analytic_jacobian_->evaluate_new_parameterization(new_parameterization);
    Matrix analytic_matrix = analytic_jacobian_->matrix();
    Matrix difference = analytic_matrix - numeric_matrix;
    return difference.max_abs() < epsilon_;
  }

  bool JacobianChecker::check_logdet(const Vector &new_parameterization) {
    Matrix numeric_matrix = numeric_jacobian_.matrix(new_parameterization);
    analytic_jacobian_->evaluate_new_parameterization(new_parameterization);
    return fabs(log(fabs(det(numeric_matrix))) - analytic_jacobian_->logdet()) <
           epsilon_;
  }

  namespace {
    class SubFunction {
     public:
      typedef std::function<Vector(const Vector &)> Mapping;
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

  bool JacobianChecker::check_second_order_elements(
      const Vector &new_parameterization,
      std::vector<std::string> *error_messages) {
    int dim = new_parameterization.size();
    std::vector<Matrix> numeric_hessians;
    for (int t = 0; t < dim; ++t) {
      SubFunction ft(inverse_transformation_, t);
      NumericalDerivatives derivatives(ft);
      numeric_hessians.push_back(derivatives.Hessian(new_parameterization));
    }

    for (int r = 0; r < dim; ++r) {
      for (int s = 0; s < dim; ++s) {
        for (int t = 0; t < dim; ++t) {
          double analytic_second_derivative =
              analytic_jacobian_->second_order_element(r, s, t);
          if (fabs(analytic_second_derivative - numeric_hessians[t](r, s)) >
              epsilon_) {
            if (error_messages) {
              std::ostringstream err;
              err << "Element (" << r << "," << s << "," << t << ")"
                  << " had a numeric second derivative of "
                  << numeric_hessians[t](r, s)
                  << " but an analytic second derivative of "
                  << analytic_second_derivative << "." << std::endl;
              error_messages->push_back(err.str());
            }
            return false;
          }
        }
      }
    }
    return true;
  }

  // The default implementation of transform_second_order_gradient is
  // correct, but slow.  Check that the method used in the concrete
  // class matches the answer you get from the reliable, slow method
  // from the base class.  Note, the reliable, slow method depends on
  // the correctness of analytic_jacobian_->second_order_element().
  bool JacobianChecker::check_transform_second_order_gradient(
      const Vector &new_parameterization, std::string *error_message) {
    int dim = new_parameterization.size();
    analytic_jacobian_->evaluate_new_parameterization(new_parameterization);
    Vector original_gradient(dim);
    original_gradient.randomize();
    SpdMatrix working_hessian(dim);
    working_hessian.randomize();

    Vector original_gradient_copy(original_gradient);
    SpdMatrix working_hessian_copy(working_hessian);

    analytic_jacobian_->transform_second_order_gradient(working_hessian,
                                                        original_gradient);
    analytic_jacobian_->Jacobian::transform_second_order_gradient(
        working_hessian_copy, original_gradient_copy);

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
      explicit LogDet(const std::shared_ptr<Jacobian> &analytic_jacobian)
          : analytic_jacobian_(analytic_jacobian) {}

      double operator()(const Vector &new_parameterization) {
        analytic_jacobian_->evaluate_new_parameterization(new_parameterization);
        return analytic_jacobian_->logdet();
      }

     private:
      std::shared_ptr<Jacobian> analytic_jacobian_;
    };
  }  // namespace

  bool JacobianChecker::check_logdet_gradient(
      const Vector &new_parameterization) {
    analytic_jacobian_->evaluate_new_parameterization(new_parameterization);
    Vector analytic_gradient = new_parameterization * 0;
    analytic_jacobian_->add_logdet_gradient(analytic_gradient);

    LogDet analytic_logdet(analytic_jacobian_);
    NumericalDerivatives derivatives(analytic_logdet);
    Vector numeric_gradient = derivatives.gradient(new_parameterization);
    return (numeric_gradient - analytic_gradient).max_abs() < epsilon_;
  }

  bool JacobianChecker::check_logdet_Hessian(
      const Vector &new_parameterization) {
    analytic_jacobian_->evaluate_new_parameterization(new_parameterization);
    int dim = new_parameterization.size();
    Matrix analytic_hessian(dim, dim, 0.0);
    analytic_jacobian_->add_logdet_Hessian(analytic_hessian);

    LogDet analytic_logdet(analytic_jacobian_);
    NumericalDerivatives derivatives(analytic_logdet);
    Matrix numeric_hessian = derivatives.Hessian(new_parameterization);
    return (numeric_hessian - analytic_hessian).max_abs() < epsilon_;
  }

}  // namespace BOOM

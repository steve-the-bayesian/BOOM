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

#include "TargetFun/Transformation.hpp"
#include <cmath>
#include "cpputil/report_error.hpp"

namespace BOOM {

  Jacobian::Jacobian() {}

  double Jacobian::logdet(const Vector &original_params) const {
    const Matrix &J(matrix(original_params));
    double det = fabs(J.det());
    if (det <= 0.0) {
      report_error("Jacobian matrix had zero determinant.");
    }
    return std::log(det);
  }

  Vector Jacobian::transform_gradient(const Vector &original_gradient,
                                      bool add_self_gradient,
                                      const Vector &original_params) {
    Vector ans = matrix(original_params) * original_gradient;
    if (add_self_gradient) {
      add_logdet_gradient(ans, original_params);
    }
    return ans;
  }

  Matrix Jacobian::transform_Hessian(const Vector &original_gradient,
                                     const Matrix &original_Hessian,
                                     bool add_self_Hessian,
                                     const Vector &original_params) {
    SpdMatrix ans = sandwich(matrix(original_params),
                             SpdMatrix(original_Hessian));
    transform_second_order_gradient(ans, original_gradient, original_params);
    if (add_self_Hessian) {
      add_logdet_Hessian(ans, original_params);
    }
    return std::move(ans);
  }

  void Jacobian::transform_second_order_gradient(
      SpdMatrix &working_hessian,
      const Vector &original_gradient,
      const Vector &original_params) {
    int dim = original_gradient.size();
    for (int r = 0; r < dim; ++r) {
      for (int s = r; s < dim; ++s) {
        for (int i = 0; i < dim; ++i) {
          working_hessian(r, s) +=
              original_gradient[i] * second_order_element(r, s, i, original_params);
        }
      }
    }
    working_hessian.reflect();
  }

  Transformation::Transformation(const Target &log_density_old_parameterization,
                                 const Mapping &inverse_mapping,
                                 Jacobian *jacobian)
      : logp_original_scale_(log_density_old_parameterization),
        inverse_mapping_(inverse_mapping),
        jacobian_(jacobian) {}

  double Transformation::operator()(const Vector &new_parameterization,
                                    Vector &gradient, Matrix &Hessian,
                                    uint nderiv) const {
    Vector original_parameterization = inverse_mapping_(new_parameterization);
    int dim = original_parameterization.size();
    Vector original_gradient;
    Matrix original_Hessian;
    if (nderiv > 0) {
      original_gradient.resize(dim);
      original_gradient = 0.0;
      if (nderiv > 1) {
        original_Hessian.resize(dim, dim);
        original_Hessian = 0.0;
      }
    }

    double ans = logp_original_scale_(
        original_parameterization, original_gradient, original_Hessian, nderiv);

    if (nderiv > 0 && !original_gradient.all_finite()) {
      report_error("Illegal values in original gradient.");
    }
    if (nderiv > 1 && !original_Hessian.all_finite()) {
      report_error("Illegal values in original Hessian.");
    }

    ans += jacobian_->logdet(original_parameterization);
    if (nderiv > 0) {
      gradient = jacobian_->transform_gradient(
          original_gradient, true, original_parameterization);
      if (!gradient.all_finite()) {
        report_error("Illegal values in transformed gradient.");
      }
      if (nderiv > 1) {
        Hessian = jacobian_->transform_Hessian(original_gradient,
                                               original_Hessian, true,
                                               original_parameterization);
        if (!Hessian.all_finite()) {
          report_error("Illegal values in transformed Hessian.");
        }
      }
    }
    return ans;
  }

  double Transformation::operator()(const Vector &new_parameterization) const {
    Vector g;
    Matrix h;
    return (*this)(new_parameterization, g, h, 0);
  }

  double Transformation::operator()(const Vector &new_parameterization,
                                    Vector &gradient) const {
    Matrix h;
    return (*this)(new_parameterization, gradient, h, 1);
  }

  double Transformation::operator()(const Vector &new_parameterization,
                                    Vector &gradient, Matrix &Hessian) const {
    return (*this)(new_parameterization, gradient, Hessian, 2);
  }

}  // namespace BOOM

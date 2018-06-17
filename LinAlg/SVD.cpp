// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "LinAlg/SVD.hpp"
#include "Eigen/SVD"
#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/EigenMap.hpp"

#include <sstream>
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace {
    using Eigen::JacobiSVD;
    using Eigen::MatrixXd;
  }  // namespace

  SingularValueDecomposition::SingularValueDecomposition(const Matrix &m)
      : singular_values_(min_dim(m)),
        left_(m.nrow(), min_dim(m)),
        right_(m.ncol(), min_dim(m)) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        EigenMap(m), Eigen::ComputeThinU | Eigen::ComputeThinV);
    EigenMap(singular_values_) = svd.singularValues();
    EigenMap(left_) = svd.matrixU();
    EigenMap(right_) = svd.matrixV();
  }

  const Vector &SingularValueDecomposition::values() const {
    return singular_values_;
  }
  const Matrix &SingularValueDecomposition::left() const { return left_; }
  const Matrix &SingularValueDecomposition::right() const { return right_; }
  Matrix SingularValueDecomposition::original_matrix() const {
    DiagonalMatrix Sigma(singular_values_);
    Matrix ans = (left_ * Sigma) * right_.transpose();
    return ans;
  }

  Matrix SingularValueDecomposition::solve(const Matrix &rhs,
                                           double tol) const {
    Matrix ans = left_.Tmult(rhs);
    for (uint i = 0; i < ans.nrow(); ++i) {
      double scale = singular_values_[i] / singular_values_[0];
      ans.row(i) *= fabs(scale) < tol ? 0 : 1.0 / singular_values_[i];
    }
    return right_ * ans;
  }

  Vector SingularValueDecomposition::solve(const Vector &rhs,
                                           double tol) const {
    Vector ans = left_.Tmult(rhs);
    for (uint i = 0; i < ans.size(); ++i) {
      double scale = singular_values_[i] / singular_values_[0];
      ans(i) *= fabs(scale) < tol ? 0 : 1.0 / singular_values_[i];
    }
    return right_ * ans;
  }

  Matrix SingularValueDecomposition::inv() const {
    bool invertible = left_.is_square() && right_.is_square() &&
                      left_.nrow() == right_.nrow();
    if (!invertible) {
      std::ostringstream err;
      err << "error in SingularValueDecomposition::inv(), only square matrices "
             "can be inverted"
          << std::endl
          << "original matrix = " << std::endl
          << original_matrix() << std::endl;
      report_error(err.str());
    }
    return solve(left_.Id());
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#include "LinAlg/Cholesky.hpp"
#include <sstream>
#include "Eigen/Cholesky"
#include "LinAlg/EigenMap.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace {
    using Eigen::MatrixXd;
  }  // namespace

  void Cholesky::decompose(const Matrix &A) {
    if (!A.is_square()) {
      pos_def_ = false;
      lower_cholesky_triangle_ = Matrix();
    } else {
      lower_cholesky_triangle_.resize(A.nrow(), A.ncol());
      Eigen::LLT<MatrixXd> eigen_cholesky(EigenMap(A));
      pos_def_ = eigen_cholesky.info() == Eigen::Success;
      if (pos_def_) {
        // If the fast version of the cholesky decomposition works, we're done!
        EigenMap(lower_cholesky_triangle_) = eigen_cholesky.matrixL();
      } else if (A.is_sym()) {
        // If the fast Cholesky decomposition failed, try a more robust version.
        Eigen::LDLT<MatrixXd> eigen_cholesky_safe(EigenMap(A));
        Vector D(A.nrow());
        EigenMap(D) = eigen_cholesky_safe.vectorD();
        EigenMap(lower_cholesky_triangle_) = eigen_cholesky_safe.matrixL();
        for (int i = 0; i < lower_cholesky_triangle_.ncol(); ++i) {
          lower_cholesky_triangle_.col(i) *= sqrt(D[i]);
        }
        EigenMap(lower_cholesky_triangle_) =
            eigen_cholesky_safe.transpositionsP().transpose() *
            EigenMap(lower_cholesky_triangle_);
      }
    }
  }

  SpdMatrix Cholesky::original_matrix() const {
    SpdMatrix ans(lower_cholesky_triangle_.nrow(), 0.0);
    ans.add_outer(lower_cholesky_triangle_);
    return ans;
  }

  SpdMatrix Cholesky::inv() const { return chol2inv(lower_cholesky_triangle_); }

  uint Cholesky::nrow() const { return lower_cholesky_triangle_.nrow(); }
  uint Cholesky::ncol() const { return lower_cholesky_triangle_.ncol(); }
  uint Cholesky::dim() const { return lower_cholesky_triangle_.nrow(); }

  void Cholesky::setL(const Matrix &L) {
    if (!L.is_square()) {
      report_error("A Cholesky triangle must be a square, lower triangular "
                   "matrix.");
    }
    lower_cholesky_triangle_ = L;
    pos_def_ = true;
  }

  Matrix Cholesky::getL(bool perform_check) const {
    if (perform_check) {
      check();
    }
    Matrix ans(lower_cholesky_triangle_);
    uint n = ans.nrow();
    for (uint i = 1; i < n; ++i) {
      std::fill(ans.col_begin(i), ans.col_begin(i) + i, 0.0);
    }
    return ans;
  }

  Matrix Cholesky::getLT() const {
    return lower_cholesky_triangle_.transpose();
  }

  // V = L LT
  // V.inv * X = LT.inv * L.inv * X
  Matrix Cholesky::solve(const Matrix &B) const {
    check();
    Matrix ans = Lsolve(lower_cholesky_triangle_, B);
    LTsolve_inplace(lower_cholesky_triangle_, ans);
    return ans;
  }

  Vector Cholesky::solve(const Vector &B) const {
    // if *this is the cholesky decomposition of A then
    // this->solve(B) = A^{-1} B.  It is NOT L^{-1} B
    check();
    Vector ans = Lsolve(lower_cholesky_triangle_, B);
    LTsolve_inplace(lower_cholesky_triangle_, ans);
    return ans;
  }

  // returns the log of the determinant of A
  double Cholesky::logdet() const {
    check();
    ConstVectorView d(diag(lower_cholesky_triangle_));
    double ans = 0;
    for (int i = 0; i < d.size(); ++i) {
      ans += std::log(fabs(d[i]));
    }
    return 2 * ans;
  }

  double Cholesky::det() const {
    check();
    ConstVectorView d(diag(lower_cholesky_triangle_));
    double ans = d.prod();
    return ans * ans;
  }

  void Cholesky::check() const {
    if (!pos_def_) {
      std::ostringstream err;
      err << "attempt to use an invalid cholesky decomposition" << std::endl
          << "lower_cholesky_triangle_ = " << std::endl
          << lower_cholesky_triangle_ << std::endl
          << "original matrix = " << std::endl
          << original_matrix();
      report_error(err.str());
    }
  }

}  // namespace BOOM

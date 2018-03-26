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
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace {
    using Eigen::MatrixXd;
    using std::cout;
    using std::endl;
  }  // namespace

  Chol::Chol(const Matrix &m)
      : lower_cholesky_triangle_(m.nrow(), m.ncol(), 0.0), pos_def_(true) {
    if (!m.is_square()) {
      pos_def_ = false;
      lower_cholesky_triangle_ = Matrix();
    } else {
      Eigen::LLT<MatrixXd> eigen_cholesky(EigenMap(m));
      pos_def_ = eigen_cholesky.info() == Eigen::Success;
      EigenMap(lower_cholesky_triangle_) = eigen_cholesky.matrixL();
    }
  }

  SpdMatrix Chol::original_matrix() const {
    SpdMatrix ans(lower_cholesky_triangle_.nrow(), 0.0);
    ans.add_outer(lower_cholesky_triangle_);
    return ans;
  }

  SpdMatrix Chol::inv() const { return chol2inv(lower_cholesky_triangle_); }

  uint Chol::nrow() const { return lower_cholesky_triangle_.nrow(); }
  uint Chol::ncol() const { return lower_cholesky_triangle_.ncol(); }
  uint Chol::dim() const { return lower_cholesky_triangle_.nrow(); }

  Matrix Chol::getL(bool perform_check) const {
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

  Matrix Chol::getLT() const {
    check();
    return lower_cholesky_triangle_.t();
  }

  // V = L LT
  // V.inv * X = LT.inv * L.inv * X
  Matrix Chol::solve(const Matrix &B) const {
    check();
    Matrix ans = Lsolve(lower_cholesky_triangle_, B);
    LTsolve_inplace(lower_cholesky_triangle_, ans);
    return ans;
  }

  Vector Chol::solve(const Vector &B) const {
    // if *this is the cholesky decomposition of A then
    // this->solve(B) = A^{-1} B.  It is NOT L^{-1} B
    check();
    Vector ans = Lsolve(lower_cholesky_triangle_, B);
    LTsolve_inplace(lower_cholesky_triangle_, ans);
    return ans;
  }

  // returns the log of the determinant of A
  double Chol::logdet() const {
    ConstVectorView d(diag(lower_cholesky_triangle_));
    double ans = 0;
    for (int i = 0; i < d.size(); ++i) {
      ans += std::log(fabs(d[i]));
    }
    return 2 * ans;
  }

  double Chol::det() const {
    ConstVectorView d(diag(lower_cholesky_triangle_));
    double ans = d.prod();
    return ans * ans;
  }

  void Chol::check() const {
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

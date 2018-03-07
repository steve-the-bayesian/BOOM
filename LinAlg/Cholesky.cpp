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

#include <LinAlg/Cholesky.hpp>
#include <sstream>
#include <cpputil/report_error.hpp>
#include <LinAlg/Vector.hpp>

extern "C" {
  /*  DPOTRF computes the Cholesky factorization of a real symmetric
   *  positive definite matrix A.
   */
  void dpotrf_(const char *, int *, double *, int *, int *);

  /*  DPOTRS solves a system of linear equations A*X = B with a symmetric
   *  positive definite matrix A using the Cholesky factorization
   *  A = U**T*U or A = L*L**T computed by DPOTRF.
   */
  void dpotrs_(const char *, int *, int *, const double *, int *, double *,
               int *, int*);

  /*  DPOTRI computes the inverse of a real symmetric positive definite
   *  matrix A using the Cholesky factorization A = U**T*U or A = L*L**T
   *  computed by DPOTRF.
   */
  void dpotri_(const char *, int *, double *, int *, int *);
}  // extern "C"

namespace BOOM {
  Chol::Chol(const Matrix &m)
      : dcmp_(m),
        pos_def_(true)
  {
    if (!m.is_square()) {
      pos_def_ = false;
      dcmp_ = Matrix();
    } else {
      int info = 0;
      int n = m.nrow();
      dpotrf_("L", &n, dcmp_.data(), &n, &info);
      if (info > 0) pos_def_ = false;
    }
  }

  SpdMatrix Chol::original_matrix() const {
    return LLT(getL(/* perform_check = */ false));
  }

  SpdMatrix Chol::inv() const {
    int n = dcmp_.nrow();
    SpdMatrix ans(dcmp_.begin(), dcmp_.end());
    int info = 0;
    dpotri_("L", &n, ans.data(), &n, &info);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < i; ++j) {
        ans(j, i) = ans(i, j);
      }
    }
    return ans;
  }

  uint Chol::nrow() const { return dcmp_.nrow();}
  uint Chol::ncol() const { return dcmp_.ncol();}
  uint Chol::dim() const { return dcmp_.nrow();}

  Matrix Chol::getL(bool perform_check) const {
    if (perform_check) {
      check();
    }
    Matrix ans(dcmp_);
    uint n = ans.nrow();
    for (uint i = 1; i < n; ++i) {
      std::fill(ans.col_begin(i), ans.col_begin(i) + i, 0.0);
    }
    return ans;
  }

  Matrix Chol::getLT() const {
    check();
    int n = nrow();
    Matrix ans(n, n, 0.0);
    for (int i = 0; i < n; ++i) {
      VectorViewIterator row(ans.row_begin(i) + i);
      ConstVectorView column(dcmp_.col(i));
      VectorViewConstIterator col(column.begin() + i);
      std::copy(col, col + n - i, row);
    }
    return(ans);
  }

  Matrix Chol::solve(const Matrix &B) const {
    check();
    Matrix ans(B);
    int n = dcmp_.nrow();
    int ncol_b = B.ncol();
    int info = 0;
    dpotrs_("L", &n, &ncol_b, dcmp_.data(), &n, ans.data(), &n, &info);
    if (info < 0) {
      report_error("Chol::solve problem with cholesky solver");
    }
    return ans;
  }

  Vector Chol::solve(const Vector &B) const {
    // if *this is the cholesky decomposition of A then
    // this->solve(B) = A^{-1} B.  It is NOT L^{-1} B
    check();
    Vector ans(B);
    int n = dcmp_.nrow();
    int ncol_b = 1;
    int info = 0;
    dpotrs_("L", &n, &ncol_b, dcmp_.data(), &n, ans.data(), &n, &info);
    if (info < 0) {
      report_error("Chol::solve problem with cholesky solver");
    }
    return ans;
  }

  // returns the log of the determinant of A
  double Chol::logdet() const {
    ConstVectorView d(diag(dcmp_));
    double ans = 0;
    for (int i = 0; i < d.size(); ++i) {
      ans += std::log(fabs(d[i]));
    }
    return 2 * ans;
  }

  double Chol::det() const {
    ConstVectorView d(diag(dcmp_));
    double ans = d.prod();
    return ans * ans;
  }

  void Chol::check() const {
    if (!pos_def_) {
      std::ostringstream err;
      err << "attempt to use an invalid cholesky decomposition" << std::endl
          << "dcmp_ = " << std::endl
          << dcmp_ << std::endl
          << "original matrix = " << std::endl
          << original_matrix();
      report_error(err.str());
    }
  }

}  // namespace BOOM

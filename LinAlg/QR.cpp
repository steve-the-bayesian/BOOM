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
#include "LinAlg/QR.hpp"
#include <cstring>
#include "Eigen/QR"
#include "LinAlg/EigenMap.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  using Eigen::HouseholderQR;
  using Eigen::Lower;
  using Eigen::MatrixXd;
  using Eigen::Upper;
  using std::cout;
  using std::endl;

  QR::QR(const Matrix &mat) { decompose(mat); }

  Matrix QR::getQ() const { return Q_; }

  Matrix QR::getR() const { return R_; }

  Matrix QR::solve(const Matrix &B) const {
    Matrix ans = Usolve(R_, B);
    EigenMap(ans) = EigenMap(Q_).transpose() * EigenMap(ans);
    return ans;
  }

  Vector QR::Qty(const Vector &y) const {
    if (length(y) != Q_.nrow()) {
      report_error("Wrong size argument y passed to QR::Qty.");
    }
    Vector ans(Q_.ncol());
    EigenMap(ans) = EigenMap(Q_).transpose() * EigenMap(y);
    return ans;
  }

  Matrix QR::QtY(const Matrix &Y) const {
    Matrix ans(Q_.ncol(), Y.ncol());
    EigenMap(ans) = EigenMap(Q_).transpose() * EigenMap(Y);
    return ans;
  }

  Vector QR::solve(const Vector &B) const {
    Vector Rinverse_B = Usolve(R_, B);
    Vector ans(Q_.nrow());
    EigenMap(ans) = EigenMap(Q_).transpose() * EigenMap(Rinverse_B);
    return ans;
  }

  double QR::det() const { return R_.diag().prod(); }

  void QR::decompose(const Matrix &mat) {
    Q_ = Matrix(mat.nrow(), mat.ncol());
    R_ = Matrix(mat.ncol(), mat.ncol(), 0.0);

    Eigen::HouseholderQR<MatrixXd> eigen_qr(EigenMap(mat));

    // Temporary is needed because you can't take the block() of a view.
    MatrixXd eigen_R = eigen_qr.matrixQR().triangularView<Upper>();
    EigenMap(R_) = eigen_R.block(0, 0, ncol(), ncol());

    // The Q matrix is stored as a vector of rotations, which logically make a
    // matrix.  We can recover that matrix by applying them to a correctly
    // shaped identity matrix.  Eigen's Identity class doesn't inherit from
    // MatrixBase, so it does not have the needed applyOnTheLeft member.  Thus
    // we work with a dense identity matrix.
    //
    // The name thin_Q is because we expect nrow() > ncol() in most settings.
    // In the full QR decomposition 'fat_Q' will be square with dimension =
    // max(nrow, ncol).
    MatrixXd thin_Q(mat.nrow(), mat.ncol());
    thin_Q.setIdentity();
    thin_Q.applyOnTheLeft(eigen_qr.householderQ());

    EigenMap(Q_) = thin_Q;
  }

  void QR::clear() {
    Q_ = Matrix(0, 0);
    R_ = Matrix(0, 0);
  }

  Vector QR::Rsolve(const Vector &Qty) const {
    assert(Qty.size() == R_.nrow());
    Vector ans = Usolve(R_, Qty);
    EigenMap(ans) = EigenMap(Q_).transpose() * EigenMap(ans);
    return ans;
  }

  Matrix QR::Rsolve(const Matrix &QtY) const { return Usolve(R_, QtY); }

  Vector QR::vectorize() const {
    Vector ans(2);
    ans[0] = nrow();
    ans[1] = ncol();
    ans.concat(ConstVectorView(Q_.data(), Q_.size(), 1));
    ans.concat(ConstVectorView(R_.data(), R_.size(), 1));
    return ans;
  }

  const double *QR::unvectorize(const double *dp) {
    int nrow = lround(*dp);
    ++dp;
    int ncol = lround(*dp);
    ++dp;
    Q_.resize(nrow, ncol);
    memcpy(Q_.data(), dp, nrow * ncol * sizeof(double));
    dp += (nrow * ncol);

    R_.resize(ncol, ncol);
    memcpy(R_.data(), dp, ncol * ncol * sizeof(double));
    dp += ncol * ncol;
    return dp;
  }

}  // namespace BOOM

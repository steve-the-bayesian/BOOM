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

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "Eigen/Cholesky"
#include "Eigen/Core"
#include "LinAlg/Eigen.hpp"
#include "LinAlg/EigenMap.hpp"
#include "LinAlg/Selector.hpp"

namespace BOOM {
  using Eigen::MatrixXd;
  namespace {
    typedef std::vector<double> dVector;
  }  // namespace

  SpdMatrix::SpdMatrix() {}

  SpdMatrix::SpdMatrix(uint dim, double x) : Matrix(dim, dim) {
    if (dim > 0) set_diag(x);
  }

  SpdMatrix::SpdMatrix(uint n, const double *x, bool ColMajor)
      : Matrix(n, n, x, ColMajor) {}

  SpdMatrix::SpdMatrix(const Vector &v, bool minimal) {
    if (v.empty()) return;
    size_t dimension = 0;
    if (minimal) {
      dimension = lround((-1 + sqrt(1 + 8 * v.size())) / 2.0);
      if (dimension * (dimension + 1) != 2 * v.size()) {
        report_error("Wrong size Vector argument to SpdMatrix constructor.");
      }
    } else {
      dimension = lround(sqrt(v.size()));
      if (dimension * dimension != v.size()) {
        report_error("Wrong size Vector argument to SpdMatrix constructor.");
      }
    }
    this->resize(dimension);
    unvectorize(v, minimal);
  }

  SpdMatrix::SpdMatrix(const Matrix &A, bool check) : Matrix(A) {
    if (check) {
      double d;
      uint imax, jmax;
      std::tie(d, imax, jmax) = A.distance_from_symmetry();
      if (d > .5) {
        std::ostringstream err;
        err << "Non-symmetric matrix passed to SpdMatrix constructor."
            << std::endl << A;
        report_error(err.str());
      } else if (d > 1e-9) {
        fix_near_symmetry();
      }
      // If the distance from symmetry is less than 1e-9 then ignore it.
    }
  }

  SpdMatrix::SpdMatrix(const SubMatrix &rhs, bool check) {
    if (check && (rhs.nrow() != rhs.ncol())) {
      report_error(
          "SpdMatrix constructor was supplied a non-square"
          "SubMatrix argument");
    }
    operator=(rhs);
  }

  SpdMatrix::SpdMatrix(const ConstSubMatrix &rhs, bool check) {
    if (check && rhs.nrow() != rhs.ncol()) {
      report_error(
          "SpdMatrix constructor was supplied a non-square"
          "SubMatrix argument");
    }
    operator=(rhs);
  }

  SpdMatrix &SpdMatrix::operator=(const SubMatrix &rhs) {
    if (rhs.nrow() != rhs.ncol()) {
      report_error(
          "SpdMatrix::operator= called with rectangular "
          "RHS argument");
    }
    Matrix::operator=(rhs);
    fix_near_symmetry();
    return *this;
  }

  SpdMatrix &SpdMatrix::operator=(const ConstSubMatrix &rhs) {
    if (rhs.nrow() != rhs.ncol()) {
      report_error(
          "SpdMatrix::operator= called with rectangular "
          "RHS argument");
    }
    Matrix::operator=(rhs);
    fix_near_symmetry();
    return *this;
  }

  SpdMatrix &SpdMatrix::operator=(const Matrix &rhs) {
    double d;
    uint imax, jmax;
    std::tie(d, imax, jmax) = rhs.distance_from_symmetry();
    if (d > .5) {
      report_error("Argument to SpdMatrix is non-symmetric.");
    }
    Matrix::operator=(rhs);
    fix_near_symmetry();
    return *this;
  }

  SpdMatrix &SpdMatrix::operator=(double x) {
    Matrix::operator=(x);
    return *this;
  }

  bool SpdMatrix::operator==(const SpdMatrix &rhs) const {
    return Matrix::operator==(rhs);
  }

  void SpdMatrix::swap(SpdMatrix &rhs) { Matrix::swap(rhs); }

  SpdMatrix &SpdMatrix::randomize(RNG &rng) {
    *this = 0.0;
    SpdMatrix tmp(nrow());
    tmp.Matrix::randomize(rng);
    EigenMap(*this).selfadjointView<Eigen::Upper>().rankUpdate(
        EigenMap(tmp).transpose(), 1.0);
    reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::randomize_gaussian(double mean, double sd, RNG &rng) {
    report_error("randomize_gaussian doesn't make sense for an SpdMatrix.  "
                 "Consider just calling randomize() instead.");
    return *this;
  }

  uint SpdMatrix::nelem() const {
    uint n = nrow();
    return n * (n + 1) / 2;
  }

  SpdMatrix &SpdMatrix::resize(uint n) {
    Matrix::resize(n, n);
    return *this;
  }

  SpdMatrix &SpdMatrix::set_diag(double x, bool zero) {
    Matrix::set_diag(x, zero);
    return *this;
  }

  SpdMatrix &SpdMatrix::set_diag(const Vector &v, bool zero) {
    Matrix::set_diag(v, zero);
    return *this;
  }

  inline void zero_upper(SpdMatrix &V) {
    uint n = V.nrow();
    for (uint i = 0; i < n; ++i) {
      dVector::iterator b = V.col_begin(i);
      dVector::iterator e = b + i;
      std::fill(b, e, 0.0);
    }
  }

  Matrix SpdMatrix::chol() const {
    bool ok = true;
    return chol(ok);
  }
  Matrix SpdMatrix::chol(bool &ok) const {
    Cholesky cholesky(*this);
    if (!cholesky.is_pos_def()) {
      ok = false;
      return Matrix(0, 0);
    } else {
      ok = true;
      return cholesky.getL(false);
    }
  }

  SpdMatrix SpdMatrix::inv() const {
    bool ok = true;
    SpdMatrix ans = inv(ok);
    if (!ok) {
      std::ostringstream err;
      err << "Matrix not positive definite...\n"
          << *this
          << "\nEigenvalues...\n"
          << eigenvalues(*this)
          << "\n";
      report_error(err.str());
    }
    return ans;
  }

  SpdMatrix SpdMatrix::inv(bool &ok) const {
    Cholesky cholesky(*this);
    if (!cholesky.is_pos_def()) {
      ok = false;
      return SpdMatrix(0);
    } else {
      ok = true;
      return cholesky.inv();
    }
  }

  double SpdMatrix::invert_inplace() {
    Eigen::LLT<Eigen::MatrixXd> eigen_chol(
        Eigen::Ref<Eigen::MatrixXd>(EigenMap(*this)));
    Eigen::MatrixXd L = eigen_chol.matrixL();
    double ans = 0;
    for (int i = 0; i < nrow(); ++i) {
      ans -= 2 * std::log(fabs(L(i, i)));
    }
    EigenMap(*this) =
        eigen_chol.solve(Eigen::MatrixXd::Identity(nrow(), nrow()));
    return ans;
  }

  double SpdMatrix::det() const {
    Cholesky L(*this);
    if (L.is_pos_def()) {
      return std::exp(L.logdet());
    } else {
      return Matrix::det();
    }
  }

  double SpdMatrix::logdet() const {
    bool ok(true);
    return logdet(ok);
  }

  double SpdMatrix::logdet(bool &ok) const {
    ok = true;
    uint n = nrow();
    if (n == 0) {
      return negative_infinity();
    } else if (n == 1) {
      double x = data()[0];
      if (x <= 0) {
        ok = false;
        return negative_infinity();
      } else {
        return std::log(x);
      }
    } else if (n == 2) {
      const double *values(data());
      // If the matrix needs to reflect then the upper triangle is
      // current, but the lower triangle might not be.  In that case
      // prefer looking at values[2], the upper-right element, over
      // values[1], the lower left element.
      double determinant = values[0] * values[3] - values[2] * values[2];
      if (determinant <= 0) {
        ok = false;
        return negative_infinity();
      }
      return std::log(determinant);
    } else {
      Matrix L(chol(ok));
      if (!ok) return BOOM::negative_infinity();
      double ans = 0.0;
      for (uint i = 0; i < n; ++i) ans += std::log(L(i, i));
      ans *= 2;
      return ans;
    }
  }

  Matrix SpdMatrix::solve(const Matrix &rhs) const {
    if (rhs.nrow() != this->ncol()) {
      report_error(
          "Number of rows in rhs does not match the number of columns "
          "in the SpdMatrix.");
    }
    Cholesky cholesky(*this);
    if (!cholesky.is_pos_def()) {
      ostringstream msg;
      msg << "Matrix not positive definite in SpdMatrix::solve(Matrix)"
          << std::endl
          << *this << std::endl;
      report_error(msg.str());
    }
    return cholesky.solve(rhs);
  }

  Vector SpdMatrix::solve(const Vector &rhs) const {
    bool ok = true;
    Vector ans(this->solve(rhs, ok));
    if (!ok) {
      ostringstream msg;
      msg << "Matrix not positive definite in SpdMatrix::solve(Vector)."
          << std::endl;
      report_error(msg.str());
    }
    return ans;
  }

  Vector SpdMatrix::solve(const Vector &rhs, bool &ok) const {
    if (rhs.size() != this->ncol()) {
      report_error("The dimensions of the matrix and vector don't match.");
    }
    Cholesky cholesky(*this);
    ok = cholesky.is_pos_def();
    if (!ok) {
      return Vector(rhs.size(), negative_infinity());
    } else {
      return cholesky.solve(rhs);
    }
  }

  void SpdMatrix::reflect() {
    uint n = nrow();
    for (uint i = 0; i < n; ++i) {
      col(i) = row(i);
    }
  }

  void SpdMatrix::fix_near_symmetry() {
    for (int i = 0; i < nrow(); ++i) {
      for (int j = 0; j < i; ++j) {
        double value = .5 * (unchecked(i, j) + unchecked(j, i));
        unchecked(i, j) = unchecked(j, i) = value;
      }
    }
  }

  double SpdMatrix::Mdist(const Vector &x, const Vector &y) const {
    return Mdist(x - y);
  }

  double SpdMatrix::Mdist(const Vector &x) const {
    int n = x.size();
    if (n != nrow()) {
      report_error("Wrong size x passed to SpdMatrix::Mdist");
    }
    const double *xdata(x.data());
    const double *thisdata(data());
    double ans = 0;
    for (int j = 0; j < n; ++j) {
      ans += xdata[j] * xdata[j] * thisdata[INDX(j, j)];
      for (int i = j + 1; i < n; ++i) {
        ans += 2 * xdata[j] * xdata[i] * thisdata[INDX(i, j)];
      }
    }
    return ans;
  }

  namespace {
    template <class V>
    void add_outer_impl(SpdMatrix &S, const V &v, double w) {
      assert(v.size() == S.nrow());
      if (S.nrow() == 0) return;
      EigenMap(S).selfadjointView<Eigen::Upper>().rankUpdate(EigenMap(v), w);
    }

    template <class VECTOR>
    void add_outer_subset_impl(SpdMatrix &S, const VECTOR &v,
                              double weight, const Selector &inc) {
      assert(S.nrow() == v.size());
      assert(inc.nvars_possible() == v.size());
      if (inc.nvars_possible() == inc.nvars()) {
        add_outer_impl(S, v, weight);
      } else {
        for (int i = 0; i < inc.nvars(); ++i) {
          int I = inc.indx(i);
          for (int j = i; j < inc.nvars(); ++j) {
            int J = inc.indx(j);
            S(I, J) += weight * v[I] * v[J];
          }
        }
      }
    }

  }  // namespace

  SpdMatrix &SpdMatrix::add_outer(const Vector &v, double w, bool force_sym) {
    add_outer_impl<Vector>(*this, v, w);
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_outer(const Vector &v, const Selector &inc,
                                  double weight, bool force_sym) {
    add_outer_subset_impl(*this, v, weight, inc);
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_outer(const VectorView &v, double w,
                                  bool force_sym) {
    add_outer_impl<VectorView>(*this, v, w);
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_outer(const VectorView &v, const Selector &inc,
                                  double weight, bool force_sym) {
    add_outer_subset_impl(*this, v, weight, inc);
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_outer(const ConstVectorView &v, double w,
                                  bool force_sym) {
    add_outer_impl<ConstVectorView>(*this, v, w);
    if (force_sym) reflect();
    return *this;
  }
  SpdMatrix &SpdMatrix::add_outer(const ConstVectorView &v, const Selector &inc,
                                  double weight, bool force_sym) {
    add_outer_subset_impl(*this, v, weight, inc);
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_outer(const Matrix &X, double w, bool force_sym) {
    if (X.nrow() == 0 || X.ncol() == 0) return *this;
    if (X.nrow() != this->nrow()) {
      report_error("Wrong number of rows in add_outer.");
    }
    EigenMap(*this).selfadjointView<Eigen::Upper>().rankUpdate(EigenMap(X), w);
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_inner(const Matrix &X, const Vector &w,
                                  bool force_sym) {
    assert(X.nrow() == w.size());
    assert(X.ncol() == this->ncol());
    uint n = w.size();
    for (uint i = 0; i < n; ++i) {
      this->add_outer(X.row(i), w[i], false);
    }
    if (force_sym) reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_inner(const Matrix &x, double w) {
    int n = nrow();
    assert(x.ncol() == this->nrow());
    uint k = x.nrow();
    if (n == 0 || k == 0) return *this;
    EigenMap(*this).selfadjointView<Eigen::Upper>().rankUpdate(
        EigenMap(x).transpose(), w);
    reflect();
    return *this;
  }

  SpdMatrix &SpdMatrix::add_inner2(const Matrix &A, const Matrix &B, double w) {
    // adds w*(A^TB + B^TA)
    assert(A.ncol() == B.ncol() && A.ncol() == nrow());
    assert(A.nrow() == B.nrow());
    if (nrow() == 0) return *this;
    EigenMap(*this) += w * (EigenMap(A).transpose() * EigenMap(B) +
                            EigenMap(B).transpose() * EigenMap(A));
    return *this;
  }

  SpdMatrix &SpdMatrix::add_outer2(const Matrix &A, const Matrix &B, double w) {
    // adds w*(AB^T + BA^T)
    assert(A.nrow() == B.nrow() && B.nrow() == nrow());
    assert(B.ncol() == A.ncol());
    if (nrow() == 0) return *this;
    EigenMap(*this) += w * (EigenMap(A) * EigenMap(B).transpose() +
                            EigenMap(B) * EigenMap(A).transpose());
    return *this;
  }

  SpdMatrix &SpdMatrix::add_outer2(const Vector &x, const Vector &y, double w) {
    assert(x.size() == nrow() && y.size() == ncol());
    if (nrow() == 0) return *this;
    EigenMap(*this).selfadjointView<Eigen::Upper>().rankUpdate(
        EigenMap(x), EigenMap(y), w);
    reflect();
    return *this;
  }

  //-------------- multiplication --------------------

  SpdMatrix &SpdMatrix::scale_off_diagonal(double scale) {
    size_t dim = nrow();
    double *el = data();
    for (size_t i = 0; i < dim; ++i) {
      for (size_t j = 0; j < dim; ++j) {
        if (i != j) {
          *el *= scale;
        }
        ++el;
      }
    }
    return *this;
  }

  //---------- general_Matrix ---------
  Matrix &SpdMatrix::mult(const Matrix &B, Matrix &ans, double scal) const {
    assert(can_mult(B, ans));
    uint m = nrow();
    uint n = B.ncol();
    if (n == 0 || m == 0) return ans;
    EigenMap(ans) =
        EigenMap(*this).selfadjointView<Eigen::Upper>() * EigenMap(B) * scal;
    return ans;
  }

  Matrix &SpdMatrix::Tmult(const Matrix &B, Matrix &ans, double scal) const {
    return mult(B, ans, scal);
  }

  Matrix &SpdMatrix::multT(const Matrix &B, Matrix &ans, double scal) const {
    return Matrix::multT(B, ans, scal);
  }

  //---------- SpdMatrix ---------
  Matrix &SpdMatrix::mult(const SpdMatrix &B, Matrix &ans, double scal) const {
    const Matrix &A(B);
    return mult(A, ans, scal);
  }

  Matrix &SpdMatrix::Tmult(const SpdMatrix &B, Matrix &ans, double scal) const {
    const Matrix &A(B);
    return Tmult(A, ans, scal);
  }

  Matrix &SpdMatrix::multT(const SpdMatrix &B, Matrix &ans, double scal) const {
    const Matrix &A(B);
    return multT(A, ans, scal);
  }

  //--------- DiagonalMatrix this and B are both symmetric ---------
  Matrix &SpdMatrix::mult(const DiagonalMatrix &B, Matrix &ans,
                          double scal) const {
    return Matrix::mult(B, ans, scal);
  }
  Matrix &SpdMatrix::Tmult(const DiagonalMatrix &B, Matrix &ans,
                           double scal) const {
    return Matrix::mult(B, ans, scal);
  }
  Matrix &SpdMatrix::multT(const DiagonalMatrix &B, Matrix &ans,
                           double scal) const {
    return Matrix::mult(B, ans, scal);
  }

  //--------- Vector --------------

  Vector &SpdMatrix::mult(const Vector &v, Vector &ans, double scal) const {
    assert(ans.size() == nrow());
    if (size() == 0) return ans;
    EigenMap(ans) =
        EigenMap(*this).selfadjointView<Eigen::Upper>() * EigenMap(v);
    return ans;
  }

  Vector &SpdMatrix::Tmult(const Vector &v, Vector &ans, double scal) const {
    return mult(v, ans, scal);
  }

  Vector SpdMatrix::vectorize(bool minimal) const {  // copies upper triangle
    uint n = ncol();
    uint ans_size = minimal ? nelem() : n * n;
    Vector ans(ans_size);
    Vector::iterator it = ans.begin();
    for (uint i = 0; i < n; ++i) {
      dVector::const_iterator b = col_begin(i);
      dVector::const_iterator e = minimal ? b + i + 1 : b + n;
      it = std::copy(b, e, it);
    }
    return ans;
  }

  void SpdMatrix::unvectorize(const Vector &x, bool minimal) {
    Vector::const_iterator b(x.begin());
    unvectorize(b, minimal);
  }

  namespace {
    template <class ITERATOR>
    ITERATOR unvectorize_impl(SpdMatrix *matrix, ITERATOR &b, bool minimal) {
      int n = matrix->ncol();
      for (int i = 0; i < n; ++i) {
        ITERATOR e = minimal ? b + i + 1 : b + n;
        std::copy(b, e, matrix->col_begin(i));
        b = e;
      }
      matrix->reflect();
      return b;
    }
  }  // namespace

  Vector::const_iterator SpdMatrix::unvectorize(Vector::const_iterator &b,
                                                bool minimal) {
    return unvectorize_impl(this, b, minimal);
  }

  ConstVectorView::const_iterator SpdMatrix::unvectorize(
      ConstVectorView::const_iterator b, bool minimal) {
    return unvectorize_impl(this, b, minimal);
  }

  void SpdMatrix::make_symmetric(bool have_upper) {
    uint n = ncol();
    for (uint i = 1; i < n; ++i) {
      for (uint j = 0; j < i; ++j) {  // (i, j) is in the lower triangle
        if (have_upper)
          unchecked(i, j) = unchecked(j, i);
        else
          unchecked(j, i) = unchecked(i, j);
      }
    }
  }

  // ================== non member functions ===========================
  SpdMatrix Id(uint p) {
    SpdMatrix ans(p);
    ans.set_diag(1.0);
    return ans;
  }

  SpdMatrix outer(const Vector &v) {
    SpdMatrix ans(v.size(), 0.0);
    ans.add_outer(v);
    return ans;
  }
  SpdMatrix outer(const VectorView &v) {
    SpdMatrix ans(v.size(), 0.0);
    ans.add_outer(v);
    return ans;
  }
  SpdMatrix outer(const ConstVectorView &v) {
    SpdMatrix ans(v.size(), 0.0);
    ans.add_outer(v);
    return ans;
  }

  SpdMatrix LLT(const Matrix &L, double a) {
    SpdMatrix ans(L.nrow(), 0.0);
    ans.add_outer(L, a, true);
    return ans;
  }

  SpdMatrix RTR(const Matrix &R, double a) {
    SpdMatrix ans(R.ncol(), 0.0);
    ans.add_inner(R, a);
    return ans;
  }

  Matrix chol(const SpdMatrix &S) { return S.chol(); }
  Matrix chol(const SpdMatrix &S, bool &ok) { return S.chol(ok); }

  SpdMatrix chol2inv(const Matrix &L) {
    assert(L.is_square());
    int n = L.nrow();
    SpdMatrix ans(n, 1.0);
    EigenMap(L).triangularView<Eigen::Lower>().solveInPlace(EigenMap(ans));
    EigenMap(L).triangularView<Eigen::Lower>().transpose().solveInPlace(
        EigenMap(ans));
    return ans;
  }

  SpdMatrix sandwich(const Matrix &A, const SpdMatrix &V) {
    if (A.size() == 0 || V.size() == 0) {
      return SpdMatrix(0);
    }
    SpdMatrix ans(A.nrow());
    EigenMap(ans) = EigenMap(A) * EigenMap(V).selfadjointView<Eigen::Upper>() *
                    EigenMap(A).transpose();
    return ans;
  }

  SpdMatrix sandwich(const Matrix &A, const Vector &diagonal) {
    DiagonalMatrix d(diagonal);
    return A.Tmult(d * A);
  }

  SpdMatrix sandwich_transpose(const Matrix &A, const Vector &diagonal) {
    Matrix tmp(A * DiagonalMatrix(diagonal));
    return(tmp.multT(A));
  }

  SpdMatrix self_diagonal_average(const SpdMatrix &X,
                                  double diagonal_shrinkage) {
    SpdMatrix ans(X);
    self_diagonal_average_inplace(ans, diagonal_shrinkage);
    return ans;
  }

  void self_diagonal_average_inplace(SpdMatrix &X,
                                     double diagonal_shrinkage) {
    if (diagonal_shrinkage < 0.0 || diagonal_shrinkage > 1.0) {
      report_error("The diagonal_shrinkage argument must be between 0 and 1.");
    }
    X.scale_off_diagonal(1 - diagonal_shrinkage);
  }

  SpdMatrix as_symmetric(const Matrix &A) {
    assert(A.is_square());
    Matrix ans = A.transpose();
    ans += A;
    ans /= 2.0;
    return SpdMatrix(ans, false);  // no symmetry check needed
  }

  SpdMatrix sum_self_transpose(const Matrix &A) {
    assert(A.is_square());
    uint n = A.nrow();
    SpdMatrix ans(n, 0.0);
    for (uint i = 0; i < n; ++i) {
      for (uint j = 0; j < i; ++j) {
        ans(i, j) = ans(j, i) = A(i, j) + A(j, i);
      }
    }
    return ans;
  }

  Vector eigenvalues(const SpdMatrix &X) {
    SymmetricEigen eigen(X, false);
    return eigen.eigenvalues();
  }

  Vector eigen(const SpdMatrix &X, Matrix &Z) {
    SymmetricEigen eigen(X, true);
    Z = eigen.eigenvectors();
    return eigen.eigenvalues();
  }

  double largest_eigenvalue(const SpdMatrix &X) { return max(eigenvalues(X)); }

  SpdMatrix operator*(double x, const SpdMatrix &V) {
    SpdMatrix ans(V);
    ans *= x;
    return ans;
  }

  SpdMatrix operator*(const SpdMatrix &V, double x) { return x * V; }

  SpdMatrix operator/(const SpdMatrix &v, double x) { return v * (1.0 / x); }

  SpdMatrix symmetric_square_root(const SpdMatrix &V) {
    Matrix eigenvectors(V.nrow(), V.nrow());
    Vector eigenvalues = eigen(V, eigenvectors);
    // We want Q^T Lambda^{1/2} Q.  We can get there by taking
    // Lambda^1/4 and pre-multiplying rows of Q.
    for (int i = 0; i < nrow(eigenvectors); ++i) {
      eigenvectors.col(i) *= sqrt(sqrt(eigenvalues[i]));
    }
    return eigenvectors.outer();
  }

  Matrix eigen_root(const SpdMatrix &X) {
    Matrix eigenvectors(X.nrow(), X.nrow());
    Vector eigenvalues = eigen(X, eigenvectors);
    for (int i = 0; i < nrow(eigenvectors); ++i) {
      eigenvectors.col(i) *= sqrt(eigenvalues[i]);
    }
    return eigenvectors.transpose();
  }

  SpdMatrix Kronecker(const SpdMatrix &A, const SpdMatrix &B) {
    uint dima = A.nrow();
    uint dimb = B.nrow();
    SpdMatrix ans(dima * dimb);
    for (int i = 0; i < dima; ++i) {
      for (int j = i; j < dima; ++j) {
        block(ans, i, j, dimb, dimb) = A(i, j) * B;
      }
    }
    ans.reflect();
    return ans;
  }

  SpdMatrix block_diagonal_spd(const std::vector<SpdMatrix> &blocks) {
    size_t total_dim = 0;
    for (const auto &el : blocks) {
      total_dim += el.nrow();
    }
    SpdMatrix ans(total_dim, 0.0);

    size_t start = 0;
    for (const auto &el : blocks) {
      SubMatrix view(ans,
                     start, start + el.nrow() - 1,
                     start, start + el.ncol() - 1);
      view = el;
      start += el.nrow();
    }
    return ans;
  }

}  // namespace BOOM

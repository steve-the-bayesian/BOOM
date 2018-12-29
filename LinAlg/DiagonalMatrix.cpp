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
#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  typedef DiagonalMatrix DM;

  DM::DiagonalMatrix() {}

  DM::DiagonalMatrix(uint n, double x) : diagonal_elements_(n, x) {}

  DM::DiagonalMatrix(const Vector &v) : diagonal_elements_(v) {}

  DM::DiagonalMatrix(const VectorView &diagonal_elements)
      : diagonal_elements_(diagonal_elements) {}

  DM::DiagonalMatrix(const ConstVectorView &diagonal_elements)
      : diagonal_elements_(diagonal_elements) {}

  DM::DiagonalMatrix(const std::vector<double> &diagonal_elements)
      : diagonal_elements_(diagonal_elements) {}

  DiagonalMatrix &DiagonalMatrix::operator=(double x) {
    diagonal_elements_ = x;
    return *this;
  }

  bool DM::operator==(const DiagonalMatrix &rhs) const {
    return diagonal_elements_ == rhs.diagonal_elements_;
  }

  void DM::swap(DM &rhs) {
    std::swap(diagonal_elements_, rhs.diagonal_elements_);
  }

  void DM::randomize() { diagonal_elements_.randomize(); }

  DM &DM::resize(uint n) {
    diagonal_elements_.resize(n);
    return *this;
  }

  VectorView DM::diag() { return VectorView(diagonal_elements_); }

  const Vector &DM::diag() const { return diagonal_elements_; }

  //---------------  Matrix multiplication -----------

  Matrix &DM::mult(const Matrix &B, Matrix &ans, double scalar) const {
    assert(nrow() == B.nrow());
    ans = B;
    for (uint i = 0; i < ans.nrow(); ++i) {
      ans.row(i) *= diagonal_elements_[i] * scalar;
    }
    return ans;
  }

  Matrix &DM::Tmult(const Matrix &B, Matrix &ans, double scal) const {
    return this->mult(B, ans, scal);
  }

  Matrix DM::Tmult(const Matrix &rhs) const {
    if (rhs.nrow() != this->nrow()) {
      report_error("Incompatible matrices in DiagonalMatrix::Tmult.");
    }
    Matrix ans(nrow(), rhs.ncol());
    Tmult(rhs, ans, 1.0);
    return ans;
  }
  
  Matrix &DM::multT(const Matrix &B, Matrix &ans, double scal) const {
    assert(ncol() == B.nrow());
    ans.resize(B.ncol(), B.nrow());
    for (uint i = 0; i < nrow(); ++i) {
      ans.row(i) = B.col(i) * diagonal_elements_[i] * scal;
    }
    return ans;
  }

  //------ SpdMatrix (this and spd both symmetric) ----------

  Matrix &DM::mult(const SpdMatrix &S, Matrix &ans, double scal) const {
    const Matrix &tmp(S);
    return this->mult(tmp, ans, scal);
  }

  Matrix &DM::Tmult(const SpdMatrix &S, Matrix &ans, double scal) const {
    const Matrix &tmp(S);
    return this->mult(tmp, ans, scal);
  }

  Matrix &DM::multT(const SpdMatrix &S, Matrix &ans, double scal) const {
    const Matrix &tmp(S);
    return this->mult(tmp, ans, scal);
  }

  void DM::sandwich_inplace(SpdMatrix &m) const {
    assert((nrow() == m.nrow()) && (m.ncol() == ncol()));
    for (int i = 0; i < nrow(); ++i) {
      m.row(i) *= diagonal_elements_[i];
      m.col(i) *= diagonal_elements_[i];
    }
  }

  SpdMatrix DM::sandwich(const SpdMatrix &m) const {
    SpdMatrix ans(m);
    sandwich_inplace(ans);
    return ans;
  }

  //------ DiagonalMatrix (this and spd both symmetric) ----------

  DiagonalMatrix &DM::mult(const DiagonalMatrix &S, DiagonalMatrix &ans,
                           double scal) const {
    ans.resize(ncol());
    ans.diag() = diagonal_elements_ * S.diagonal_elements_;
    if (scal != 1.0) {
      ans.diagonal_elements_ *= scal;
    }
    return ans;
  }

  DiagonalMatrix &DM::Tmult(const DiagonalMatrix &S, DiagonalMatrix &ans,
                            double scal) const {
    return mult(S, ans, scal);
  }

  DiagonalMatrix &DM::multT(const DiagonalMatrix &S, DiagonalMatrix &ans,
                            double scal) const {
    return mult(S, ans, scal);
  }

  //---------- Vector ------------
  Vector &DM::mult(const Vector &v, Vector &ans, double scal) const {
    ans = diagonal_elements_ * v;
    if (scal != 1.0) {
      ans *= scal;
    }
    return ans;
  }

  Vector &DM::Tmult(const Vector &v, Vector &ans, double scal) const {
    return this->mult(v, ans, scal);
  }

  namespace {
    template <class VECTOR>
    Vector mult_impl(const DiagonalMatrix &mat,
                     const VECTOR &v) {
      if (v.size() != mat.ncol()) {
        report_error("Vector is incompatible with diagonal matrix.");
      }

      Vector ans(mat.nrow(), 0.0);
      const ConstVectorView &diagonal(mat.diag());
      for (int i = 0; i < mat.ncol(); ++i) {
        ans[i] = v[i] * diagonal[i];
      }
      return ans;
    }

    template <class VECTOR>
    void in_place_multiplication(const Vector &diag, VECTOR &v) {
      if (diag.size() != v.size()) {
        report_error("wrong size argument for in_place_multiplication.");
      }
      for (int i = 0; i < v.size(); ++i) {
        v[i] *= diag[i];
      }
    }
  }  // namespace 

  void DM::multiply_inplace(Vector &v) const {
    in_place_multiplication(diagonal_elements_, v);
  }

  void DM::multiply_inplace(VectorView &v) const {
    in_place_multiplication(diagonal_elements_, v);
  }
  
  Vector DM::operator*(const Vector &v) const {
    return mult_impl(*this, v);
  }
  Vector DM::operator*(const VectorView &v) const {
    return mult_impl(*this, v);
  }
  Vector DM::operator*(const ConstVectorView &v) const {
    return mult_impl(*this, v);
  }
  
  DiagonalMatrix DM::t() const { return *this; }

  DiagonalMatrix DM::inv() const {
    DiagonalMatrix ans(1.0 / diagonal_elements_);
    return ans;
  }

  DiagonalMatrix DM::inner() const {
    return DiagonalMatrix(diagonal_elements_ * diagonal_elements_);
  }

  Matrix DM::solve(const Matrix &mat) const { return inv() * mat; }

  Vector DM::solve(const Vector &v) const { return v / diagonal_elements_; }

  double DM::det() const { return prod(); }

  double DM::logdet() const {
    double ans = 0;
    for (auto el : diagonal_elements_) ans += log(el);
    return ans;
  }
  
  ostream &DM::print(ostream &out) const {
    Matrix tmp(nrow(), ncol(), 0.0);
    tmp.diag() = diagonal_elements_;
    return out << tmp;
  }

  Vector DM::singular_values() const {
    Vector ans(diag());
    std::sort(ans.begin(), ans.end(), std::greater<double>());
    return ans;
  }

  Vector DM::real_evals() const {
    Vector ans(diag());
    std::sort(ans.begin(), ans.end(), std::greater<double>());
    return ans;
  }

  DM &DM::operator+=(double x) {
    diagonal_elements_ += x;
    return *this;
  }

  DM &DM::operator+=(const DiagonalMatrix &rhs) {
    diagonal_elements_ += rhs.diagonal_elements_;
    return *this;
  }

  DM &DM::operator-=(double x) {
    diagonal_elements_ -= x;
    return *this;
  }

  DM &DM::operator-=(const DiagonalMatrix &rhs) {
    diagonal_elements_ -= rhs.diagonal_elements_;
    return *this;
  }

  DM &DM::operator*=(double x) {
    diagonal_elements_ *= x;
    return *this;
  }

  DM &DM::operator*=(const DiagonalMatrix &rhs) {
    diagonal_elements_ *= rhs.diagonal_elements_;
    return *this;
  }

  DM &DM::operator/=(const DiagonalMatrix &rhs) {
    diagonal_elements_ /= rhs.diagonal_elements_;
    return *this;
  }

  DM &DM::operator/=(double x) {
    diagonal_elements_ /= x;
    return *this;
  }

  double DM::sum() const { return diagonal_elements_.sum(); }

  double DM::prod() const { return diagonal_elements_.prod(); }

  DiagonalMatrix operator-(const DiagonalMatrix &d) {
    return DiagonalMatrix(-1 * d.diag());
  }

  DiagonalMatrix operator*(const DiagonalMatrix &m1, const DiagonalMatrix &m2) {
    DiagonalMatrix ans;
    return m1.mult(m2, ans);
  }

  Matrix operator*(const DiagonalMatrix &m1, const Matrix &m2) {
    Matrix ans;
    return m1.mult(m2, ans);
  }

  Matrix operator*(const Matrix &m1, const DiagonalMatrix &m2) {
    Matrix ans;
    return m1.mult(m2, ans);
  }

  Matrix operator*(const DiagonalMatrix &m1, const SpdMatrix &m2) {
    Matrix ans;
    return m1.mult(m2, ans);
  }

  Matrix operator*(const SpdMatrix &m1, const DiagonalMatrix &m2) {
    Matrix ans;
    return m1.mult(m2, ans);
  }

  DiagonalMatrix operator+(const DiagonalMatrix &d, double x) {
    DiagonalMatrix ans(d);
    ans += x;
    return ans;
  }

  DiagonalMatrix operator+(double x, const DiagonalMatrix &d) { return d + x; }

  DiagonalMatrix operator+(const DiagonalMatrix &m1, const DiagonalMatrix &m2) {
    DiagonalMatrix ans(m1);
    ans += m2;
    return ans;
  }

  Matrix operator+(const DiagonalMatrix &m1, const Matrix &m2) {
    Matrix ans(m2);
    ans.diag() += m1.diag();
    return ans;
  }

  Matrix operator+(const Matrix &m1, const DiagonalMatrix &m2) {
    return m2 + m1;
  }

  DiagonalMatrix operator-(const DiagonalMatrix &m1, const DiagonalMatrix &m2) {
    DiagonalMatrix ans(m1);
    ans -= m2;
    return ans;
  }

  Matrix operator-(const DiagonalMatrix &m1, const Matrix &m2) {
    Matrix ans(-m2);
    ans.diag() += m1.diag();
    return ans;
  }

  Matrix operator-(const Matrix &m1, const DiagonalMatrix &m2) {
    Matrix ans(m1);
    ans.diag() -= m2.diag();
    return ans;
  }

  DiagonalMatrix operator/(const DiagonalMatrix &m1, const DiagonalMatrix &m2) {
    DiagonalMatrix ans(m2.inv());
    ans *= m1;
    return ans;
  }

  DiagonalMatrix operator-(const DiagonalMatrix &d, double x) {
    DiagonalMatrix ans(d);
    ans -= x;
    return ans;
  }

  DiagonalMatrix operator-(double x, const DiagonalMatrix &d) {
    DiagonalMatrix ans(-d);
    ans += x;
    return ans;
  }

  DiagonalMatrix operator*(const DiagonalMatrix &d, double x) {
    DiagonalMatrix ans(d);
    ans *= x;
    return ans;
  }

  DiagonalMatrix operator*(double x, const DiagonalMatrix &d) { return d * x; }

  DiagonalMatrix operator/(const DiagonalMatrix &d, double x) {
    return d * (1.0 / x);
  }

  DiagonalMatrix operator/(double x, const DiagonalMatrix &d) {
    return d.inv() * x;
  }

}  // namespace BOOM

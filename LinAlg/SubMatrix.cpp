// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "LinAlg/SubMatrix.hpp"
#include "cpputil/report_error.hpp"
#include <iomanip>

namespace BOOM {
  using std::endl;
  namespace {
    typedef SubMatrix SM;
    typedef ConstSubMatrix CSM;
  }  // namespace
  SM::SubMatrix(Matrix &m, uint rlo, uint rhi, uint clo, uint chi)
      : start_(m.data() + rlo + clo * m.nrow()),
        nr_(rhi - rlo + 1),
        nc_(chi - clo + 1),
        stride(m.nrow()) {
    if (nr_ < 0) {
      report_error("SubMatrix number of rows can't be negative.");
    }
    if (nc_ < 0) {
      report_error("SubMatrix number of columns can't be negative.");
    }
    if (rhi >= m.nrow()) {
      std::ostringstream err;
      err << "Submatrix final row index " << rhi << " must be less than "
          << "the number of rows in the host matrix " << m.nrow() << ".";
      report_error(err.str());
    }
    if (chi >= m.ncol()) {
      std::ostringstream err;
      err << "Submatrix final column index " << chi << " must be less than "
          << "the number of columns in the host matrix " << m.ncol() << ".";
      report_error(err.str());
    }
  }

  SM::SubMatrix(Matrix &m)
      : start_(m.data()),
        nr_(m.nrow()),
        nc_(m.ncol()),
        stride(m.nrow()) {}

  SM::SubMatrix(double *v, int nrow, int ncol)
      : start_(v),
        nr_(nrow),
        nc_(ncol),
        stride(nrow) {}

  SM::SubMatrix(SM &m, uint rlo, uint rhi, uint clo, uint chi)
      : start_(m.start_ + rlo + clo * m.stride),
        nr_(rhi - rlo + 1),
        nc_(chi - clo + 1),
        stride(m.stride) {}

  SM::SubMatrix(const SM &rhs)
      : start_(rhs.start_), nr_(rhs.nr_), nc_(rhs.nc_), stride(rhs.stride) {}

  SM &SM::operator=(const SM &rhs) {
    if (rhs.nrow() != nr_ || rhs.ncol() != nc_) {
      report_error("Matrix of wrong dimension passed to assignment operator.");
    }
    for (uint i = 0; i < nc_; ++i) {
      std::copy(rhs.col_begin(i), rhs.col_end(i), cols(i));
    }
    return *this;
  }

  SM &SM::reset(const SM &rhs) {
    if (&rhs != this) {
      start_ = rhs.start_;
      nr_ = rhs.nr_;
      nc_ = rhs.nc_;
      stride = rhs.stride;
    }
    return *this;
  }

  SM & SubMatrix::reset(Matrix &rhs, int rlo, int rhi, int clo, int chi) {
    start_ = rhs.data() + rlo + clo * rhs.nrow();
    nr_ = (rhi - rlo + 1);
    nc_ = (chi - clo + 1);
    stride = (rhs.nrow());
    if (nr_ < 0) {
      report_error("rlo must be nonnegative and no larger than rhi.");
    }
    if (nc_ < 0) {
      report_error("clo must be nonnegative and no larger than chi.");
    }
    if (rhi >= rhs.nrow()) {
      report_error("rhi must be smaller than the number of rows in "
                   "the host matrix.");
    }
    if (chi >= rhs.ncol()) {
      report_error("chi must be smaller than the number of column in "
                   "the host matrix.");
    }
    return *this;
  }

  SM &SM::reset(double *data, int nrow, int ncol, int new_stride) {
    start_ = data;
    nr_ = nrow;
    nc_ = ncol;
    stride = new_stride;
    return *this;
  }

  SM &SM::operator=(const Matrix &rhs) {
    if (rhs.nrow() != nr_ || rhs.ncol() != nc_) {
      report_error("Matrix of wrong dimension passed to assignment operator.");
    }
    for (uint i = 0; i < nc_; ++i) {
      std::copy(rhs.col_begin(i), rhs.col_end(i), cols(i));
    }
    return *this;
  }

  SM &SM::operator=(double scalar) {
    for (uint i = 0; i < nc_; ++i) {
      col(i) = scalar;
    }
    return *this;
  }

  //------------------------------------------------------------
  uint SM::nrow() const { return nr_; }
  uint SM::ncol() const { return nc_; }
  //------------------------------------------------------------
  VectorView SM::col(uint j) {
    VectorView ans(cols(j), nr_, 1);
    return ans;
  }
  ConstVectorView SM::col(uint j) const {
    ConstVectorView ans(cols(j), nr_, 1);
    return ans;
  }
  VectorView SM::last_col() { return col(nc_ - 1); }
  ConstVectorView SM::last_col() const { return col(nc_ - 1); }

  //------------------------------------------------------------
  VectorView SM::row(uint i) {
    VectorView ans(cols(0) + i, nc_, stride);
    return ans;
  }
  ConstVectorView SM::row(uint i) const {
    ConstVectorView ans(cols(0) + i, nc_, stride);
    return ans;
  }
  VectorView SM::last_row() { return row(nr_ - 1); }
  ConstVectorView SM::last_row() const { return row(nr_ - 1); }

  VectorView SM::diag() {
    int m = std::min(nr_, nc_);
    return VectorView(cols(0), m, stride + 1);
  }
  ConstVectorView SM::diag() const {
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0), m, stride + 1);
  }

  VectorView SM::subdiag(int i) {
    if (i < 0) return superdiag(-i);
    int m = std::min(nr_, nc_);
    return VectorView(cols(0) + i, m - i, stride + 1);
  }

  ConstVectorView SM::subdiag(int i) const {
    if (i < 0) return superdiag(-i);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0) + i, m - i, stride + 1);
  }

  VectorView SM::superdiag(int i) {
    if (i < 0) return subdiag(-1);
    int m = std::min(nr_, nc_);
    return VectorView(cols(i), m - i, stride + 1);
  }

  ConstVectorView SM::superdiag(int i) const {
    if (i < 0) return subdiag(-1);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(i), m - i, stride + 1);
  }

  //------------------------------------------------------------
  double SM::sum() const {
    double ans = 0;
    for (uint i = 0; i < nc_; ++i) ans += col(i).sum();
    return ans;
  }
  //------------------------------------------------------------
  Matrix SM::to_matrix() const {
    Matrix ans(nrow(), ncol());
    for (int i = 0; i < ncol(); ++i) {
      ans.col(i) = this->col(i);
    }
    return ans;
  }
  //------------------------------------------------------------
  std::ostream &SM::display(std::ostream &out, int precision) const {
    ConstSubMatrix m(*this);
    return m.display(out, precision);
  }

  std::ostream &operator<<(std::ostream &out, const SubMatrix &m) {
    return m.display(out, 5);
  }

  //------------------------------------------------------------

  double &SM::operator()(uint i, uint j) {
    # ifndef NDEBUG
    if (i >= nr_ || j >= nc_) {
      report_error("Index out of bounds.");
    }
    # endif
    return cols(j)[i];
  }
  //------------------------------------------------------------
  const double &SM::operator()(uint i, uint j) const {
    # ifndef NDEBUG
    if (i >= nr_ || j >= nc_) {
      report_error("Index out of bounds.");
    }
    # endif
    return cols(j)[i];
  }
  //------------------------------------------------------------
  double *SM::col_begin(uint j) { return cols(j); }
  double *SM::col_end(uint j) { return cols(j) + nr_; }

  const double *SM::col_begin(uint j) const { return cols(j); }
  const double *SM::col_end(uint j) const { return cols(j) + nr_; }
  //------------------------------------------------------------

  SM &SM::operator+=(const Matrix &rhs) {
    assert(rhs.nrow() == nr_ && rhs.ncol() == nc_);
    for (uint i = 0; i < nc_; ++i) {
      VectorView v(cols(i), nr_, 1);
      v += rhs.col(i);
    }
    return *this;
  }

  SM &SM::operator+=(const SubMatrix &rhs) {
    assert(rhs.nrow() == nr_ && rhs.ncol() == nc_);
    for (uint i = 0; i < nc_; ++i) {
      VectorView v(cols(i), nr_, 1);
      v += rhs.col(i);
    }
    return *this;
  }

  SM &SM::operator+=(const ConstSubMatrix &rhs) {
    assert(rhs.nrow() == nr_ && rhs.ncol() == nc_);
    for (uint i = 0; i < nc_; ++i) {
      VectorView v(cols(i), nr_, 1);
      v += rhs.col(i);
    }
    return *this;
  }

  SM &SM::operator-=(const Matrix &rhs) {
    assert(rhs.nrow() == nr_ && rhs.ncol() == nc_);
    for (uint i = 0; i < nc_; ++i) {
      VectorView v(cols(i), nr_, 1);
      v -= rhs.col(i);
    }
    return *this;
  }

  SM &SM::operator-=(const SubMatrix &rhs) {
    assert(rhs.nrow() == nr_ && rhs.ncol() == nc_);
    for (uint i = 0; i < nc_; ++i) {
      VectorView v(cols(i), nr_, 1);
      v -= rhs.col(i);
    }
    return *this;
  }

  SM &SM::operator-=(const ConstSubMatrix &rhs) {
    assert(rhs.nrow() == nr_ && rhs.ncol() == nc_);
    for (uint i = 0; i < nc_; ++i) {
      VectorView v(cols(i), nr_, 1);
      v -= rhs.col(i);
    }
    return *this;
  }

  SM &SM::operator+=(double x) {
    for (uint i = 0; i < nc_; ++i) {
      col(i) += x;
    }
    return *this;
  }

  SM &SM::operator-=(double x) {
    for (uint i = 0; i < nc_; ++i) {
      col(i) -= x;
    }
    return *this;
  }

  SM &SM::operator*=(double x) {
    for (uint i = 0; i < nc_; ++i) {
      col(i) *= x;
    }
    return *this;
  }

  SM &SM::operator/=(double x) {
    for (uint i = 0; i < nc_; ++i) {
      col(i) /= x;
    }
    return *this;
  }

  //======================================================================
  CSM::ConstSubMatrix(const Matrix &m)
      : start_(m.data()), nr_(m.nrow()), nc_(m.ncol()), stride(m.nrow()) {}

  CSM::ConstSubMatrix(const SubMatrix &m)
      : start_(m.start_), nr_(m.nr_), nc_(m.nc_), stride(m.stride) {}

  CSM::ConstSubMatrix(const Matrix &m, uint rlo, uint rhi, uint clo, uint chi)
      : start_(m.data() + clo * m.nrow() + rlo),
        nr_(rhi - rlo + 1),
        nc_(chi - clo + 1),
        stride(m.nrow()) {
    if (rlo < 0 || clo < 0) {
      report_error("Row and column indices cannot be less than zero.");
    }
    if (rhi >= m.nrow()) {
      report_error("Row index exceeds maximum number of rows.");
    }
    if (chi >= m.ncol()) {
      report_error("Column index exceeds maximum number of rows.");
    }
    if (rhi < rlo) {
      report_error("Upper row index is less than lower index.");
    }
    if (chi < clo) {
      report_error("Upper column index is less than lower index.");
    }
  }

  CSM::ConstSubMatrix(const double *data, int nrow, int ncol, int my_stride)
      : start_(data),
        nr_(nrow),
        nc_(ncol),
        stride(my_stride >= 1 ? my_stride : nr_) {
    assert(nr_ >= 0);
    assert(nc_ >= 0);
    assert(stride >= 1);
  }

  CSM & ConstSubMatrix::reset(const Matrix &rhs, int rlo, int rhi,
                              int clo, int chi) {
    start_ = rhs.data() + rlo + clo * rhs.nrow();
    nr_ = (rhi - rlo + 1);
    nc_ = (chi - clo + 1);
    stride = (rhs.nrow());
    assert(nr_ >= 0);
    assert(nc_ >= 0);
    assert(rhi < rhs.nrow() && chi < rhs.ncol());
    return *this;
  }


  uint CSM::nrow() const { return nr_; }
  uint CSM::ncol() const { return nc_; }
  const double &CSM::operator()(uint i, uint j) const {
    assert(i < nr_ && j < nc_);
    return cols(j)[i];
  }
  //------------------------------------------------------------
  const double *CSM::col_begin(uint j) const { return cols(j); }
  const double *CSM::col_end(uint j) const { return cols(j) + nr_; }
  ConstVectorView CSM::col(uint j) const {
    ConstVectorView ans(cols(j), nr_, 1);
    return ans;
  }
  ConstVectorView CSM::last_col() const { return col(nc_ - 1); }
  ConstVectorView CSM::row(uint i) const {
    ConstVectorView ans(cols(0) + i, nc_, stride);
    return ans;
  }
  ConstVectorView CSM::last_row() const { return row(nr_ - 1); }

  ConstVectorView CSM::diag() const {
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0), m, stride + 1);
  }

  ConstVectorView CSM::subdiag(int i) const {
    if (i < 0) return superdiag(-i);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(0) + i, m - i, stride + 1);
  }

  ConstVectorView CSM::superdiag(int i) const {
    if (i < 0) return subdiag(-1);
    int m = std::min(nr_, nc_);
    return ConstVectorView(cols(i), m - i, stride + 1);
  }

  //------------------------------------------------------------
  double CSM::sum() const {
    double ans = 0;
    for (uint i = 0; i < nc_; ++i) ans += col(i).sum();
    return ans;
  }
  //------------------------------------------------------------
  Matrix CSM::to_matrix() const {
    Matrix ans(nrow(), ncol());
    for (int i = 0; i < ncol(); ++i) {
      ans.col(i) = this->col(i);
    }
    return ans;
  }
  //------------------------------------------------------------
  Matrix CSM::transpose() const {
    Matrix ans(ncol(), nrow());
    for (int i = 0; i < nrow(); ++i) {
      for (int j = 0; j < ncol(); ++j) {
        ans(j, i) = (*this)(i, j);
      }
    }
    return ans;
  }
  //------------------------------------------------------------
  std::ostream &CSM::display(std::ostream &out, int precision) const {
    out << std::setprecision(precision);
    for (uint i = 0; i < nrow(); ++i) {
      for (uint j = 0; j < ncol(); ++j)
        out << std::setw(8) << (*this)(i, j) << " ";
      out << endl;
    }
    return out;
  }
  //------------------------------------------------------------
  std::ostream &operator<<(std::ostream &out, const ConstSubMatrix &m) {
    return m.display(out, 5);
  }

  namespace {
    template <class M1, class M2>
    bool MatrixEquals(const M1 &lhs, const M2 &rhs) {
      if (lhs.nrow() != rhs.nrow()) return false;
      if (lhs.ncol() != rhs.ncol()) return false;
      for (int i = 0; i < lhs.nrow(); ++i) {
        for (int j = 0; j < lhs.ncol(); ++j) {
          if (lhs(i, j) != rhs(i, j)) return false;
        }
      }
      return true;
    }
  }  // namespace

  bool operator==(const Matrix &lhs, const SubMatrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const Matrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const SubMatrix &lhs, const Matrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const SubMatrix &lhs, const SubMatrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const SubMatrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const ConstSubMatrix &lhs, const Matrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const ConstSubMatrix &lhs, const SubMatrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }
  bool operator==(const ConstSubMatrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixEquals(lhs, rhs);
  }

  namespace {
    template <class M1, class M2>
    Matrix MatrixAdd(const M1 &m1, const M2 &m2) {
      Matrix ans(m1);
      SubMatrix view(ans);
      view += m2;
      return ans;
    }
    template <class MAT>
    Matrix MatrixAddScalar(const MAT &m, double x) {
      Matrix ans(m);
      ans += x;
      return ans;
    }

  }  // namespace

  Matrix operator+(const ConstSubMatrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const ConstSubMatrix &lhs, const SubMatrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const ConstSubMatrix &lhs, const Matrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const SubMatrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const SubMatrix &lhs, const SubMatrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const SubMatrix &lhs, const Matrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const Matrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const Matrix &lhs, const SubMatrix &rhs) {
    return MatrixAdd(lhs, rhs);
  }
  Matrix operator+(const ConstSubMatrix &lhs, double rhs) {
    return MatrixAddScalar(lhs, rhs);
  }
  Matrix operator+(const SubMatrix &lhs, double rhs) {
    return MatrixAddScalar(lhs, rhs);
  }
  Matrix operator+(double lhs, const ConstSubMatrix &rhs) {
    return MatrixAddScalar(rhs, lhs);
  }
  Matrix operator+(double lhs, const SubMatrix &rhs) {
    return MatrixAddScalar(rhs, lhs);
  }

  namespace {
    template <class M1, class M2>
    Matrix MatrixSubtract(const M1 &m1, const M2 &m2) {
      Matrix ans(m1);
      SubMatrix view(ans);
      view -= m2;
      return ans;
    }

    template <class MAT>
    Matrix MatrixSubtractFromScalar(double x, const MAT &m) {
      Matrix ans(m.nrow(), m.ncol(), x);
      ans -= m;
      return ans;
    }
  }  // namespace

  Matrix operator-(const ConstSubMatrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const ConstSubMatrix &lhs, const SubMatrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const ConstSubMatrix &lhs, const Matrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const SubMatrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const SubMatrix &lhs, const SubMatrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const SubMatrix &lhs, const Matrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const Matrix &lhs, const ConstSubMatrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const Matrix &lhs, const SubMatrix &rhs) {
    return MatrixSubtract(lhs, rhs);
  }
  Matrix operator-(const ConstSubMatrix &lhs, double rhs) {
    return MatrixAddScalar(lhs, -rhs);
  }
  Matrix operator-(const SubMatrix &lhs, double rhs) {
    return MatrixAddScalar(lhs, -rhs);
  }
  Matrix operator-(double lhs, const ConstSubMatrix &rhs) {
    return MatrixSubtractFromScalar(lhs, rhs);
  }
  Matrix operator-(double lhs, const SubMatrix &rhs) {
    return MatrixSubtractFromScalar(lhs, rhs);
  }

  namespace {
    template <class MAT>
    Matrix MatrixScalarMultiply(const MAT &x, double y) {
      Matrix ans(x);
      ans *= y;
      return ans;
    }

    template <class MAT1, class MAT2>
    Matrix MatrixElementDivide(const MAT1 &x, const MAT2 &y) {
      Matrix ans(x);
      ans /= y;
      return ans;
    }

    template <class MAT>
    Matrix ScalarDivideMatrix(double x, const MAT &m) {
      Matrix ans(m.nrow(), m.ncol(), x);
      ans /= m;
      return ans;
    }

  }  // namespace

  Matrix operator*(const ConstSubMatrix &x, double y) {
    return MatrixScalarMultiply(x, y);
  }
  Matrix operator*(const SubMatrix &x, double y) {
    return MatrixScalarMultiply(x, y);
  }
  Matrix operator*(double x, const ConstSubMatrix &y) {
    return MatrixScalarMultiply(y, x);
  }
  Matrix operator*(double x, const SubMatrix &y) {
    return MatrixScalarMultiply(y, x);
  }

  Matrix operator/(const ConstSubMatrix &x, const ConstSubMatrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const ConstSubMatrix &x, const SubMatrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const ConstSubMatrix &x, const Matrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const SubMatrix &x, const ConstSubMatrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const SubMatrix &x, const SubMatrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const SubMatrix &x, const Matrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const Matrix &x, const ConstSubMatrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const Matrix &x, const SubMatrix &y) {
    return MatrixElementDivide(x, y);
  }
  Matrix operator/(const ConstSubMatrix &x, double y) {
    return x * (1.0 / y);
  }
  Matrix operator/(const SubMatrix &x, double y) {
    return x * (1.0 / y);
  }
  Matrix operator/(double x, const ConstSubMatrix &y) {
    return ScalarDivideMatrix(x, y);
  }
  Matrix operator/(double x, const SubMatrix &y) {
    return ScalarDivideMatrix(x, y);
  }

  SubMatrix block(Matrix &m, int block_row, int block_col,
                  int block_row_size, int block_col_size) {
    int first_row = block_row_size * block_row;
    int last_row = first_row + block_row_size - 1;
    int first_col = block_col_size * block_col;
    int last_col = first_col + block_col_size - 1;
    return SubMatrix(m, first_row, last_row, first_col, last_col);
  }

  ConstSubMatrix const_block(const Matrix &m, int block_row, int block_col,
                             int block_row_size, int block_col_size) {
    int first_row = block_row_size * block_row;
    int last_row = first_row + block_row_size - 1;
    int first_col = block_col_size * block_col;
    int last_col = first_col + block_col_size - 1;
    return ConstSubMatrix(m, first_row, last_row, first_col, last_col);
  }

}  // namespace BOOM

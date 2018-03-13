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
#ifndef BOOM_SUBMATRIX_HPP
#define BOOM_SUBMATRIX_HPP
#include <LinAlg/Matrix.hpp>
namespace BOOM{
  // A Rectangular view into a matrix specified by lower
  // and upper coordinates (inclusive)
  class SubMatrix{
   public:
    typedef double * col_iterator;
    typedef const double * const_col_iterator;

    SubMatrix(double *v = 0, int nrow = 0, int ncol = 0);
    SubMatrix(Matrix &, uint rlo, uint rhi, uint clo, uint chi);
    SubMatrix(SubMatrix &, uint rlo, uint rhi, uint clo, uint chi);
    SubMatrix(const SubMatrix &rhs);
    explicit SubMatrix(Matrix &); // view into the entire matrix
    SubMatrix & operator=(const Matrix &rhs);

    // copy values from rhs to *this, but they stay in distinct
    // memory.  operator= with a SubMatrix rhs should work the same
    // as operator=(const Matrix &rhs)
    SubMatrix & operator=(const SubMatrix &rhs);

    // Pointer semantics: make the memory here point to the memory
    // there.
    SubMatrix & reset(const SubMatrix &rhs);
    SubMatrix & reset(double *v, int nrow, int ncol, int stride);

    uint nrow()const;
    uint ncol()const;

    double & operator()(uint i, uint j);
    const double & operator()(uint i, uint j)const;

    col_iterator col_begin(uint j);
    const_col_iterator col_begin(uint j)const;

    col_iterator col_end(uint j);
    const_col_iterator col_end(uint j)const;

    VectorView col(uint j);
    ConstVectorView col(uint j)const;
    VectorView last_col();
    ConstVectorView last_col()const;

    VectorView row(uint j);
    ConstVectorView row(uint j)const;
    VectorView last_row();
    ConstVectorView last_row()const;

    VectorView diag();
    ConstVectorView diag()const;
    VectorView subdiag(int i);
    ConstVectorView subdiag(int i)const;
    VectorView superdiag(int i);
    ConstVectorView superdiag(int i)const;

    SubMatrix & operator+=(const Matrix &m);
    SubMatrix & operator+=(const SubMatrix &m);
    SubMatrix & operator+=(const ConstSubMatrix &m);

    SubMatrix & operator-=(const Matrix &m);
    SubMatrix & operator-=(const SubMatrix &m);
    SubMatrix & operator-=(const ConstSubMatrix &m);

    SubMatrix & operator*=(double x);
    SubMatrix & operator/=(double x);
    SubMatrix & operator+=(double x);
    SubMatrix & operator-=(double x);

    double sum()const;

    Matrix to_matrix()const;
    ostream & display(ostream &out, int precision)const;

   private:
    double *start_;
    uint nr_, nc_;  // number of rows and columns in the SubMatrix
    uint stride;    // number of rows in the parent matrix
    double *cols(int i){return start_ + stride*i;}
    const double *cols(int i)const{return start_ + stride*i;}

    friend class ConstSubMatrix;
  };
  ostream & operator<<(ostream &out, const SubMatrix &m);

  //======================================================================
  // A view into a rectangular subset of a matrix.
  class ConstSubMatrix{
   public:
    typedef const double * const_col_iterator;
    ConstSubMatrix(const Matrix &);
    ConstSubMatrix(const SubMatrix &);

    // Args:
    //   m:  The matrix to be subsetted.
    //   rlo:  The row number of the first row in the subset.
    //   rhi:  The row number of the last row in the subset.
    //   clo:  The column number of the first column in the subset.
    //   chi:  The column number of the last column in the subset.
    ConstSubMatrix(const Matrix &m, uint rlo, uint rhi, uint clo, uint chi);

    // Args:
    //   data: The data for the matrix elements in column major order.
    //   rows:  The number of rows in the matrix.
    //   cols:  The number of columns in the matrix.
    //   stride: The number of memory steps needed to get to the
    //     same row, next column.  A negative entry for stride in
    //     this constructor will set 'stride' = 'rows', so in most
    //     cases the stride argument can be ignored.
    ConstSubMatrix(const double *data, int rows, int cols, int stride = -1);

    uint nrow()const;
    uint ncol()const;

    const double & operator()(uint i, uint j)const;
    const_col_iterator col_begin(uint j)const;
    const_col_iterator col_end(uint j)const;

    // TODO(stevescott):  range checking
    ConstVectorView col(uint j)const;
    ConstVectorView last_col()const;
    ConstVectorView row(uint j)const;
    ConstVectorView last_row()const;
    ConstVectorView diag()const;
    ConstVectorView subdiag(int i)const;
    ConstVectorView superdiag(int i)const;

    double sum()const;

    Matrix to_matrix()const;
    ostream & display(ostream &out, int precision)const;

   private:
    const double *start_;
    uint nr_, nc_;
    uint stride;
    const double *cols(int i)const{return start_ + stride*i;}
  };

  ostream & operator<<(ostream &out, const ConstSubMatrix &m);
  bool operator==(const Matrix &lhs, const SubMatrix &rhs);
  bool operator==(const Matrix &lhs, const ConstSubMatrix &rhs);
  bool operator==(const SubMatrix &lhs, const Matrix &rhs);
  bool operator==(const SubMatrix &lhs, const SubMatrix &rhs);
  bool operator==(const SubMatrix &lhs, const ConstSubMatrix &rhs);
  bool operator==(const ConstSubMatrix &lhs, const Matrix &rhs);
  bool operator==(const ConstSubMatrix &lhs, const SubMatrix &rhs);
  bool operator==(const ConstSubMatrix &lhs, const ConstSubMatrix &rhs);

  Matrix operator+(const ConstSubMatrix &lhs, const ConstSubMatrix &rhs);
  Matrix operator+(const SubMatrix &lhs, const ConstSubMatrix &rhs);
  Matrix operator+(const Matrix &lhs, const ConstSubMatrix &rhs);
  Matrix operator+(const ConstSubMatrix &lhs, const SubMatrix &rhs);
  Matrix operator+(const ConstSubMatrix &lhs, const Matrix &rhs);

  Matrix operator-(const ConstSubMatrix &lhs, const ConstSubMatrix &rhs);
  Matrix operator-(const SubMatrix &lhs, const ConstSubMatrix &rhs);
  Matrix operator-(const Matrix &lhs, const ConstSubMatrix &rhs);
  Matrix operator-(const ConstSubMatrix &lhs, const SubMatrix &rhs);
  Matrix operator-(const ConstSubMatrix &lhs, const Matrix &rhs);

}  // namespace BOOM

#endif // BOOM_SUBMATRIX_HPP

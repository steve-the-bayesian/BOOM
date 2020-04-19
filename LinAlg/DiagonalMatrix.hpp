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
#ifndef BOOM_NEWLA_DIAGONAL_MATRIX_HPP
#define BOOM_NEWLA_DIAGONAL_MATRIX_HPP
#include <iosfwd>
#include <vector>
#include "LinAlg/Matrix.hpp"

namespace BOOM {
  using std::istream;
  using std::ostream;

  class Vector;
  class VectorView;
  class SpdMatrix;
  // A DiagonalMatrix is logically a square Matrix, but it has a different
  // implementation.  The diagonal elements are stored in a Vector, and you
  // cannot set off-diagonal elements.  The only way to set diagonal elements is
  // to call the diag() function to access them, and then set them as you would
  // a Vector.
  class DiagonalMatrix {
   public:
    DiagonalMatrix();
    explicit DiagonalMatrix(uint dimension, double diagonal_elements = 0.0);
    explicit DiagonalMatrix(const Vector &diagonal_elements);
    explicit DiagonalMatrix(const VectorView &diagonal_elements);
    explicit DiagonalMatrix(const ConstVectorView &diagonal_elements);
    explicit DiagonalMatrix(const std::vector<double> &diagonal_elements);

    DiagonalMatrix &operator=(const double value);

    bool operator==(const DiagonalMatrix &) const;

    void swap(DiagonalMatrix &rhs);

    // fills entries with U(0,1) random variables.
    void randomize();

    // size queries
    uint nrow() const { return diagonal_elements_.size(); }
    uint ncol() const { return diagonal_elements_.size(); }

    //---- change size and shape  -----
    DiagonalMatrix &resize(uint n);

    // Diagonal elements.  Returns a VectorView instead of a Vector &
    // to prevent resizing.
    VectorView diag();
    const Vector &diag() const;

    //====== Linear algebra.  Lots of special cases here. ==================

    //------- Matrix
    // Fill 'ans' with scalar * this * B.
    // Return ans.
    Matrix &mult(const Matrix &B, Matrix &ans, double scalar = 1.0) const;

    // Fill 'ans' with scalar * this^T * B.
    // Return ans.
    Matrix &Tmult(const Matrix &B, Matrix &ans, double scalar = 1.0) const;

    Matrix Tmult(const Matrix &rhs) const;

    // Fill 'ans' with scalar * this * B^T
    // Return ans.
    Matrix &multT(const Matrix &B, Matrix &ans, double scalar = 1.0) const;

    //------- SpdMatrix

    Matrix &mult(const SpdMatrix &S, Matrix &ans, double scalar = 1.0) const;

    Matrix &Tmult(const SpdMatrix &S, Matrix &ans, double scalar = 1.0) const;

    Matrix &multT(const SpdMatrix &S, Matrix &ans, double scalar = 1.0) const;

    // this * m * this
    SpdMatrix sandwich(const SpdMatrix &m) const;
    void sandwich_inplace(SpdMatrix &m) const;

    // no BLAS support for this^T * S
    // virtual Matrix & Tmult(const SpdMatrix &S, Matrix & ans,
    //     double scalar = 1.0)const;

    //------- DiagonalMatrix
    DiagonalMatrix &mult(const DiagonalMatrix &B, DiagonalMatrix &ans,
                         double scalar = 1.0) const;
    DiagonalMatrix &Tmult(const DiagonalMatrix &B, DiagonalMatrix &ans,
                          double scalar = 1.0) const;
    DiagonalMatrix &multT(const DiagonalMatrix &B, DiagonalMatrix &ans,
                          double scalar = 1.0) const;

    //------- Vector
    Vector &mult(const Vector &v, Vector &ans, double scalar = 1.0) const;
    Vector &Tmult(const Vector &v, Vector &ans, double scalar = 1.0) const;
    void multiply_inplace(Vector &v) const;
    void multiply_inplace(VectorView &v) const;

    Vector operator*(const Vector &x) const;
    Vector operator*(const VectorView &x) const;
    Vector operator*(const ConstVectorView &x) const;

    DiagonalMatrix t() const;
    DiagonalMatrix inv() const;
    DiagonalMatrix inner() const;  // returns X^tX

    Matrix solve(const Matrix &mat) const;
    Vector solve(const Vector &v) const;
    double det() const;
    double logdet() const;
    Vector singular_values() const;  // sorted largest to smallest
    uint rank(double prop = 1e-12) const;
    // 'rank' is the number of singular values at least 'prop' times
    // the largest
    Vector real_evals() const;

    //--------  Math -------------
    DiagonalMatrix &operator+=(double x);
    DiagonalMatrix &operator+=(const DiagonalMatrix &m);

    DiagonalMatrix &operator-=(double x);
    DiagonalMatrix &operator-=(const DiagonalMatrix &m);

    DiagonalMatrix &operator*=(double x);
    DiagonalMatrix &operator*=(const DiagonalMatrix &m);

    DiagonalMatrix &operator/=(double x);
    DiagonalMatrix &operator/=(const DiagonalMatrix &m);

    double sum() const;
    double prod() const;

    ostream &print(ostream &out) const;

   private:
    BOOM::Vector diagonal_elements_;
  };

  inline ostream &operator<<(ostream &out, const DiagonalMatrix &m) {
    return m.print(out);
  }

  //--------------- Math operators --------------------
  DiagonalMatrix operator-(const DiagonalMatrix &d);

  DiagonalMatrix operator+(const DiagonalMatrix &m1, const DiagonalMatrix &m2);
  DiagonalMatrix operator+(const DiagonalMatrix &d, double x);
  DiagonalMatrix operator+(double x, const DiagonalMatrix &d);
  Matrix operator+(const DiagonalMatrix &m1, const Matrix &m2);
  Matrix operator+(const Matrix &m1, const DiagonalMatrix &m2);

  DiagonalMatrix operator-(const DiagonalMatrix &m1, const DiagonalMatrix &m2);
  DiagonalMatrix operator-(double x, const DiagonalMatrix &d);
  DiagonalMatrix operator-(const DiagonalMatrix &d, double x);
  Matrix operator-(const DiagonalMatrix &m1, const Matrix &m2);
  Matrix operator-(const Matrix &m1, const DiagonalMatrix &m2);

  DiagonalMatrix operator*(const DiagonalMatrix &m1, const DiagonalMatrix &m2);
  DiagonalMatrix operator*(const DiagonalMatrix &d, double x);
  DiagonalMatrix operator*(double x, const DiagonalMatrix &d);
  Matrix operator*(const DiagonalMatrix &m1, const Matrix &m2);
  Matrix operator*(const Matrix &m1, const DiagonalMatrix &m2);
  Matrix operator*(const DiagonalMatrix &m1, const SpdMatrix &m2);
  Matrix operator*(const SpdMatrix &m1, const DiagonalMatrix &m2);

  // pre-multiplication
  inline Vector operator*(const Vector &v, const DiagonalMatrix &m) {
    return m * v;
  }
  inline Vector operator*(const VectorView &v, const DiagonalMatrix &m) {
    return m * v;
  }
  inline Vector operator*(const ConstVectorView &v, const DiagonalMatrix &m) {
    return m * v;
  }

  DiagonalMatrix operator/(const DiagonalMatrix &m1, const DiagonalMatrix &m2);
  DiagonalMatrix operator/(const DiagonalMatrix &d, double x);
  DiagonalMatrix operator/(double x, const DiagonalMatrix &d);

}  // namespace BOOM

#endif  // BOOM_NEWLA_MATRIX_HPP

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
#ifndef BOOM_NEWLA_QR_HPP
#define BOOM_NEWLA_QR_HPP

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {
  // The QR decomposition of a matrix X.  In the case where the number
  // of rows >= the number of columns, the dimension of Q matches that
  // of X, and R is a square, upper triangular matrix with dimension
  // number_of_columns.  Q satisfies Q^T * Q = I.
  //
  // If the number of columns > number of rows, then Q is a square
  // matrix of dimension number_of_rows, and R is a trapezoidal
  // matrix of dimension matching X.
  class QR {
   public:
    // An empty QR decomposition that can be used to decompose a
    // future matrix with 'decompose', or deserialized using
    // 'unvectorize'.
    QR() {}

    // The QR decomposition of the matrix X.
    explicit QR(const Matrix &m, bool just_compute_R = false);

    // Extract the Q and R matrices from the decomposition.
    const Matrix &getQ() const {return Q_;}
    const Matrix &getR() const {return R_;}

    Matrix solve(const Matrix &B) const;
    Vector solve(const Vector &b) const;

    // Multiply the vector or matrix y by the transpose of Q.
    Vector Qty(const Vector &y) const;
    Matrix QtY(const Matrix &Y) const;

    // Multiply the argument by R inverse.
    Vector Rsolve(const Vector &Qty) const;
    Matrix Rsolve(const Matrix &QtY) const;

    // The determinant of the decomposed matrix.
    double det() const;

    // The log absolute value of the determinant of the decomposed matrix.
    double logdet() const;
    
    // Reset *this to the decomposition of the matrix m.
    // Args:
    //   m:  The matrix to decompose.
    //   just_compute_R: If 'true' then only the R matrix is computed.  In this
    //     case the only trustworthy method of this class is getR().
    void decompose(const Matrix &m, bool just_compute_R = false);

    // Reset *this to an empty state.  After a call to clear() a call
    // to decompose() or unvectorize() must be made before the object
    // can do anything useful.
    void clear();

    uint nrow() const { return Q_.nrow(); }
    uint ncol() const { return Q_.ncol(); }

    // Serialize the contents of 'this' into a Vector.
    Vector vectorize() const;

    // Resets the contents of *this from a sequence of data.  The
    // inverse operation of 'vectorize'.
    const double *unvectorize(const double *data);

   private:
    Matrix Q_;
    Matrix R_;

    // The sign of the determinant of Q.
    int sign_;
  };
  
}  // namespace BOOM
#endif  // BOOM_NEWLA_QR_HPP

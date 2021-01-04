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
#ifndef BOOM_CHOL_HPP
#define BOOM_CHOL_HPP

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

namespace BOOM {
  class Cholesky {
   public:

    // A default constructor for use when a cholesky object is needed, but the
    // matrix to be decomposed is not yet available.  It is the user's
    // responsibility to ensure that the decompose() method is called before any
    // other methods are called.
    Cholesky() : pos_def_(false) {}

    // Compute and store the Cholesky factor of the matrix 'A'.
    explicit Cholesky(const Matrix &A) { decompose(A); }

    // Compute and store the Cholesky factor of the matrix 'A'.  Any previous
    // decomposition is discarded.
    void decompose(const Matrix &A);

    // All three of these return the number of rows in the represented matrix
    // (which is the same as the number of columns).
    uint nrow() const;
    uint ncol() const;
    uint dim() const;

    // Set the lower Cholesky triangle.  L must be a square, lower triangular
    // matrix.
    void setL(const Matrix &L);

    // The lower Cholesky triangle of A.
    // If A is not positive definite then:
    //   - this matrix will not in general be triangular, because of pivoting.
    //   - the solve() and inv() methods will not work.
    Matrix getL(bool perform_check = true) const;

    // The upper Cholesky triangle of A.
    Matrix getLT() const;

    // The (inverse of A) times B.
    Matrix solve(const Matrix &B) const;

    // The (inverse of A) times b.
    Vector solve(const Vector &b) const;

    // The inverse of A.
    SpdMatrix inv() const;  // inverse of A

    // The original (represented) matrix.
    SpdMatrix original_matrix() const;

    // Determinant of A.
    double det() const;

    // Natural log of the determinant of A.
    double logdet() const;

    // Returns true if A is positive definite.  Computing a cholesky
    // decomposition is a fast way to determine if a matrix is positive
    // definite.  If the result is false, then other computations are not to be
    // trusted, and may result in errors or exceptions.
    bool is_pos_def() const { return pos_def_; }

   private:
    Matrix lower_cholesky_triangle_;
    bool pos_def_;
    void check() const;
  };

}  // namespace BOOM

#endif  // BOOM_CHOL_HPP

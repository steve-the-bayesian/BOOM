// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_LINALG_EIGEN_HPP_
#define BOOM_LINALG_EIGEN_HPP_
#include <complex>
#include <vector>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {
  // This file contains tools to buld the eigen decomposition of
  // (a) A possibly rectangular matrix, and
  // (b) A symmetric, positive definite matrix.
  //
  // The package 'Eigen' is used under the covers to do the heavy lifting, but
  // these tools are not part of the Eigen package.
  //
  // TODO: rename this file to avoid confusion with the Eigen package.

  // Eigenstructure of a square, non-symmetric matrix.
  class EigenDecomposition {
   public:
    // Args:
    //   mat: The (square) matrix for which the eigendecomposition is desired.
    //   vectors: If 'true' then both eigenvalues and eigenvectors are computed.
    //     If 'false' then only eigenvalues are computed.
    explicit EigenDecomposition(const Matrix &mat, bool vectors = true);

    // Complex conjugate eigenvalues occur consecutively.  The entry
    // with the positive imaginary part comes first.
    std::vector<std::complex<double>> eigenvalues() const {
      return eigenvalues_;
    }

    // The real and imaginary parts of all the eigenvalues.  If all
    // eigenvalues are real then the imaginary_values() will be a
    // vector of zeros (up to numerical accuracy).
    const Vector &real_eigenvalues() const { return real_eigenvalues_; }
    const Vector &imaginary_eigenvalues() const {
      return imaginary_eigenvalues_;
    }

    // Requests for eigenvectors will throw exceptions if eigenvectors
    // were not requested by the constructor.
    ConstVectorView real_eigenvector(int i) const;
    ConstVectorView imaginary_eigenvector(int i) const;
    std::vector<std::complex<double>> eigenvector(int i) const;

   private:
    std::vector<std::complex<double>> eigenvalues_;
    Vector real_eigenvalues_;
    Vector imaginary_eigenvalues_;

    // The real and imaginary parts of the (right) eigenvectors.  Each column is
    // (part of) an eigenvector.
    Matrix real_eigenvectors_;
    Matrix imaginary_eigenvectors_;
  };

  // Eigenvalues and vectors of an SpdMatrix, which will have only real
  // eigenvalues.
  class SymmetricEigen {
   public:
    // Args:
    //   matrix:  The matrix whose eigendecomposition is desired.
    //   compute_vectors: If true then eigenvalues and eigenvectors are both
    //     computed.  If false only the eigenvalues are computed.
    explicit SymmetricEigen(const SpdMatrix &matrix, bool compute_vectors = true);

    // The eigenvalues of the decomposed matrix.  The eigenvalues are in
    // increasing order so eigenvalues_[0] is smallest.
    const Vector &eigenvalues() const { return eigenvalues_; }

    // This matrix is size zero if eigenvectors were not requested.  Otherwise
    // the columns of the matrix are the right-eigenvectors.
    const Matrix &eigenvectors() const { return right_vectors_; }

    // Reconstruct the original matrix that was decomposed.
    SpdMatrix original_matrix() const;

    // The closest positive definite matrix to the original matrix.  Despite the
    // name, an SpdMatrix might not actually be positive definite.
    SpdMatrix closest_positive_definite() const;

    // The generalized_inverse is taken by inverting the nonzero eigenvalues,
    // leaving the eigenvectors the same.
    //
    // The definition of a generalized inverse (G) of a symmetric matrix A is
    // that A G A = A.  If A = E V E', where V is a diagonal matrix of
    // eigenvalues, then replacing the nonzero eigenvalues with their
    // reciprocals satisfies this requirement.
    //
    // Args:
    //   threshold: A positive real value.  After scaling by the largest
    //     eigenvalue, eigenvalues smaller than threshold are treated as zero,
    //     and not inverted.
    //
    // Returns:
    //   The generalized inverse of the decomposed matrix.
    SpdMatrix generalized_inverse(double threshold = 1e-8) const;

    // Args:
    //   threshold: A positive real value.  After scaling by the largest
    //     eigenvalue, eigenvalues smaller than threshold are treated as zero.
    //
    // Returns:
    //   The sum of the (negative) logs of the nonzero (absolute) eigenvalues.
    double generalized_inverse_logdet(double threshold = 1e-8) const;

   private:
    // Eigenvalues of the decomposed matrix.  Stored smallest to largest.
    Vector eigenvalues_;

    // The eigenvectors are the columns of right_vectors.  This matrix will be
    // of size zero if compute_vectors==false in the constructor.
    Matrix right_vectors_;
  };
}  // namespace BOOM

#endif  //  BOOM_LINALG_EIGEN_HPP_

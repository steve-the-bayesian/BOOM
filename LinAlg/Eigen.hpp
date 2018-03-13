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
#include <LinAlg/Matrix.hpp>
#include <LinAlg/Vector.hpp>
#include <complex>
#include <vector>

namespace BOOM {
  // Eigenstructure of a square, non-symmetric matrix.
  class Eigen {
   public:
    // The constructor will generate all eigenvalues of the matrix
    // 'mat'.
    Eigen(const Matrix &mat,
          bool right_vectors = false,
          bool left_vectors = false);

    // Complex conjugate eigenvalues occur consecutively.  The entry
    // with the positive imaginary part comes first.
    std::vector<std::complex<double> > eigenvalues()const;

    // The real and imaginary parts of all the eigenvalues.  If all
    // eigenvalues are real then the imaginary_values() will be a
    // vector of zeros (up to numerical accuracy).
    const Vector &real_eigenvalues()const;
    const Vector &imaginary_eigenvalues()const;

    // Requests for eigenvectors will throw exceptions if eigenvectors
    // were not requested by the constructor.
    const ConstVectorView right_real_eigenvector(int i)const;
    Vector right_imaginary_eigenvector(int i)const;
    std::vector<std::complex<double> > right_eigenvector(int i) const;

    // Indicates whether eigenvalue i is part of a conjugate pair
    // relationship.  If imaginary_sign(i) == 0 then the eigenvalue is
    // fully real.  If imaginary_sign(i) == 1 then the eigenvalue is
    // part of a conjugate pair, the imaginary component is positive
    // and eigenvalue i+1 is its conjugate.  If imaginary_sign(i) ==
    // -1 then the imaginary part is negative, and the eigenvalue is
    // conjugate to eigenvalue i-1.
    int imaginary_sign(int i)const;
   private:
    Vector real_eigenvalues_;
    Vector imaginary_eigenvalues_;

    // Entry i of imaginary_sign_ is 0 if eigenvalue i is entirely
    // real, 1 if part of a conjugate pair with a positive imaginary
    // component, and -1 if part of a conjugate pair with a negative
    // imaginary component.  If imaginary_sign_[i] == -1 then
    // imaginary_sign_[i-1] == 1.
    std::vector<int> imaginary_sign_;
    Vector zero_;

    // If requested by the constructor, the eigenvectors are stored in
    // the columns of left_vectors_ or right_vectors_.  The order is
    // the same as the eigenvalues.
    //
    // If eigenvalues j and j+1 form a conjugate pair then the j'th
    // vector is v[, j] + i*v[, j+1], and the j+1'st eigenvector is
    // v[, j] - i*v[, j+1].
    Matrix left_vectors_;
    Matrix right_vectors_;

  };
}

#endif //  BOOM_LINALG_EIGEN_HPP_

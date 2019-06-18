#ifndef BOOM_LINALG_LU_HPP_
#define BOOM_LINALG_LU_HPP_
/*
  Copyright (C) 2005-2019 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

#include <memory>

namespace BOOM {

  // The LU implementation depends on Eigen.  It is hidden to avoid Eigen
  // leaking into the BOOM source code.
  namespace LuImpl {
    class LU_impl_;
  }  // namespace LuImpl;

  
  // The LU decomposition of a square matrix A.  The decomposition involves
  // pivoting for numerical stability, so the L and U matrices are not made
  // directly available.
  class LU {
   public:
    // Set impl_ to nullptr.
    LU();
    
    // Args:
    //   square_matrix:  The matrix to be decomposed.
    explicit LU(const Matrix &square_matrix);

    // The 'rule of 5' members are trivial, but they must be defined because the
    // type of impl_ is hidden.
    ~LU();
    LU(const LU &rhs);
    LU(LU &&rhs);
    LU & operator=(const LU &rhs);
    LU & operator=(LU &&rhs);

    // Construct and store the LU decomposition of A.
    void decompose(const Matrix &mat);

    // Recompute the original matrix.
    Matrix original_matrix() const;

    // square_matrix.inv() * rhs
    Vector solve(const ConstVectorView &rhs) const;
    Matrix solve(const Matrix &rhs) const;

    // The number of rows and columns in A.
    int nrow() const;
    int ncol() const;

    // Set impl_ to nullptr.
    void clear();

    // The determinant of A.
    double det() const;

    // The log of the determinant of A.  If the determinant is negative this
    // will produce negative infinity.
    double logdet() const;

   private:
    std::unique_ptr<LuImpl::LU_impl_> impl_;
    
  };
  
}  // namespace BOOM


#endif  //  BOOM_LINALG_LU_HPP_

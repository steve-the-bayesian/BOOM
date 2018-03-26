// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_SVD_HPP
#define BOOM_SVD_HPP
#include <limits>
#include "LinAlg/Matrix.hpp"

namespace BOOM {

  class SingularValueDecomposition {
   public:
    explicit SingularValueDecomposition(const Matrix &m);
    const Vector &values() const;
    const Matrix &left() const;
    const Matrix &right() const;

    Matrix original_matrix() const;
    Matrix solve(const Matrix &RHS,
                 double tol = std::numeric_limits<double>::epsilon()) const;

    Vector solve(const Vector &RHS,
                 double tol = std::numeric_limits<double>::epsilon()) const;

    Matrix inv() const;  // inverse of the original matrix, if square

   private:
    // Return the smaller of the the number of rows, or the number of columns,
    // in the Matrix m.
    int min_dim(const Matrix &m) const { return std::min(m.nrow(), m.ncol()); }

    // the SVD is A = U S V.t()
    Vector singular_values_;  // diagonal of S

    // Left singular vectors are the columns of left_;
    Matrix left_;  // U

    // Right singular vectors are the columns of right_;
    Matrix right_;  // V.t()
  };
}  // namespace BOOM
#endif  // BOOM_SVD_HPP

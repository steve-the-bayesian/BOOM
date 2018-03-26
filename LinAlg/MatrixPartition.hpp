// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

#ifndef BOOM_MATRIX_PARTITION_HPP_
#define BOOM_MATRIX_PARTITION_HPP_

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

namespace BOOM {
  class MatrixPartition {
   public:
    MatrixPartition(Matrix *m, const std::vector<int> &row_sizes,
                    const std::vector<int> &col_sizes);
    SubMatrix operator()(int i, int j);

    // Returns a view into the (i,j) blo
    const SubMatrix operator()(int i, int j) const;

    // If premultiply == true then this returns a view into the
    // portion of vector v that would be multiplied by row block i
    // in the multiplication m * v.  If premultiply == false then it
    // returns a view into the portion of v that would be multiplied
    // by column block i in the multiplication v * m.
    VectorView view(Vector &v, int i, bool premultiply = true) const;
    VectorView view(VectorView v, int i, bool premultiply = true) const;

    void reset(Matrix *m);

   private:
    Matrix *m_;
    std::vector<int> row_start_;
    std::vector<int> col_start_;
    int row_max_;
    int col_max_;
  };

}  // namespace BOOM
#endif  // BOOM_MATRIX_PARTITION_HPP_

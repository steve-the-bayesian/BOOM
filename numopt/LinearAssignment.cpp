/*
  Copyright (C) 2005-2021 Steven L. Scott

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

#include "numopt/LinearAssignment.hpp"
#include "numopt/linear_assignment/lap.hpp"

namespace BOOM {

  double LinearAssignment::solve() {
    int dim = cost_matrix_.nrow();
    row_solution_.resize(dim);
    col_solution_.resize(dim);
    Vector row_dual_variables(dim);
    Vector col_dual_variables(dim);

    Matrix cost = cost_matrix_.transpose();

    return lap(
        cost.nrow(),
        cost.data(),
        row_solution_.data(),
        col_solution_.data(),
        row_dual_variables.data(),
        col_dual_variables.data());
  }


}  // namespace BOOM

#ifndef BOOM_NUMOPT_LINEAR_ASSIGNMENT_HPP_
#define BOOM_NUMOPT_LINEAR_ASSIGNMENT_HPP_
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

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {

  // A solver for the linear assignment problem.  Given a square matrix of costs
  // describing how much it will cost row[i] to perform the task in column j,
  // and under the constraint that all tasks must be performed by exactly one
  // row, find the permutation of the indices {0, ..., dim - 1} (call it ans),
  // so that the minimum costs is achieved if row j performs task ans[j].
  class LinearAssignment {
   public:

    // Args:
    //   cost_matrix: Element (i, j) is the cost if row i performs task j.
    explicit LinearAssignment(const Matrix &cost_matrix)
        : cost_matrix_(cost_matrix)
    {}

    // Args:
    //   assignment: The total cost of assigning task assignment[i] to worker i
    //     (summed over i).
    double cost(const std::vector<int> &assignment) const {
      double ans = 0;
      for (size_t i = 0; i < assignment.size(); ++i) {
        ans += cost_matrix_(i, assignment[i]);
      }
      return ans;
    }

    // Find the optimal solution.
    double solve();

    // The optimal assignment of tasks to workers.  Assigning task
    // row_solution[j] to worker j minimizes cost.
    const std::vector<long> &row_solution() const {return row_solution_;}

    // The optimal assignment of workers to tasks.  Assigning task j to
    // col_solution[j] minimizes costs.
    const std::vector<long> &col_solution() const {return col_solution_;}

   private:
    Matrix cost_matrix_;
    std::vector<long> row_solution_;
    std::vector<long> col_solution_;
  };

}  // namespace BOOM
#endif // BOOM_NUMOPT_LINEAR_ASSIGNMENT_HPP_

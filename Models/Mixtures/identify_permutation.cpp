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

#include "Models/Mixtures/identify_permutation.hpp"
#include "cpputil/seq.hpp"
#include "cpputil/report_error.hpp"
#include "LinAlg/Matrix.hpp"


namespace BOOM {

  namespace {
    Matrix compute_mean_cluster_probs(
        const std::vector<Matrix> &cluster_probs,
        const std::vector<std::vector<int>> &permutation) {
      int niter = cluster_probs.size();
      if (niter <= 0) {
        report_error("Cluster probabilities must include at least 1 iteration.");
      }

      int nobs = cluster_probs[0].nrow();
      int nclusters = cluster_probs[0].ncol();

      Matrix ans(nobs, nclusters, 0.0);
      for (int i = 0; i < niter; ++i) {
        for (int k = 0; k < nclusters; ++k) {
          ans.col(k) += cluster_probs[i].col(permutation[i][k]);
        }
      }
      return ans / niter;
    }

    // Args:
    //   probs: A nobs x nclusters matrix of cluster probabilities as assigned
    //     by a previously run MCMC algorithm.
    //   mean_probs: A nobs x nclusters matrix of estimated cluster
    //     probabilities.
    //   permutation: Output.  The permutation minimizing the KL divergence
    //     between probs and mean_probs.
    double solve_linear_assignment_problem(
        const Matrix &probs,
        const Matrix &mean_probs,
        std::vector<int> &permuation) {
      int nclusters = mean_probs.ncol();
      Matrix log_mean_probs = log(mean_probs);
      Matrix log_probs = log(probs);
      Matrix cost(nclusters, nclusters);

      for (int i = 0; i < nclusters; ++i) {
        for (int j = 0; j < nclusters; ++j) {
          cost(i, j) = sum(probs.col(j) * (
              log_probs.col(j) - log_mean_probs.col(i)));
        }
      }
      LinearAssignment lap(cost);

      double min_cost = lap.solve();
      permuation = std::vector<int>(lap.row_solution().begin(),
                                    lap.row_solution().end());
      return min_cost;
    }
  }  // namespace

  std::vector<std::vector<int>>
  identify_permutation(const std::vector<Matrix> &cluster_probs) {
    int niter = cluster_probs.size();
    int nclusters = cluster_probs[0].ncol();

    std::vector<std::vector<int>> permutation;
    for (int i = 0; i < niter; ++i) {
      permutation.push_back(seq<int>(0, nclusters - 1));
    }

    double total_cost = infinity();
    double cost_reduction = infinity();
    while (cost_reduction > 1e-5) {
      Matrix mean_cluster_probs = compute_mean_cluster_probs(
          cluster_probs, permutation);
      double old_total_cost = total_cost;
      total_cost = 0;
      for (int draw = 0; draw < niter; ++draw) {
        total_cost += solve_linear_assignment_problem(
            cluster_probs[draw],
            mean_cluster_probs,
            permutation[draw]);
      }
      cost_reduction = old_total_cost - total_cost;
    }
    return permutation;
  }

}  // namespace BOOM

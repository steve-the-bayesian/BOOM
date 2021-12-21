#ifndef BOOM_MODELS_MIXTURES_IDENTIFY_PERMUTATION_HPP_
#define BOOM_MODELS_MIXTURES_IDENTIFY_PERMUTATION_HPP_
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

#include <vector>

#include "LinAlg/Array.hpp"
#include "LinAlg/Matrix.hpp"
#include "numopt/LinearAssignment.hpp"

namespace BOOM {

  // Given a set of MCMC draws of membership probabilities, return a permutation
  // of state labels for each MCMC draw.
  //
  // Args:
  //   Array: Element (i, j, k) gives the probability that unit j belongs to
  //   cluster k in Monte Carlo iteration i.
  //
  // Returns:
  //   Return element (i j) is the new cluster label for the cluster that had
  //   been labeled 'j' in Monte Carlo iteration i.
  std::vector<std::vector<int>> identify_permutation_from_probs(
      const Array &cluster_probs);
  std::vector<std::vector<int>> identify_permutation_from_probs(
      const std::vector<Matrix> &cluster_probs);

  // Given a set of MCMC draws of cluster indicators, return a permutation of
  // state labels for each MCMC draw that attempts to remove label switching.
  //
  // Args:
  //   cluster_labels: Element (i, j) is the cluster label for observation j in
  //     Monte Carlo iteration i.  Each iteration should contain the labels
  //     0..K-1, and K should not change from iteration to iteration.
  //
  // Returns:
  //   Return element (i j) is the new cluster label for the cluster that had
  //   been labeled 'j' in Monte Carlo iteration i.
  std::vector<std::vector<int>> identify_permutation_from_labels(
      const std::vector<std::vector<int>> &cluster_labels);
}

#endif //  BOOM_MODELS_MIXTURES_IDENTIFY_PERMUTATION_HPP_

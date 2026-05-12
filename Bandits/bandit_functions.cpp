/*
  Copyright (C) 2005-2026 Steven L. Scott

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

#include "Bandits/bandit_functions.hpp"
#include "stats/optimal_arm_probabilities.hpp"

namespace BOOM {
  Vector ComputeOptimalArmProbabilities(const Matrix &values, RNG &rng) {
    return compute_optimal_arm_probabilities(values, rng);
  }

  Vector ValueRemainingDistribution(const Matrix &values, RNG &rng) {
    int number_arms = values.ncol();
    std::vector<int> workspace(number_arms);

    Vector arm_probs = ComputeOptimalArmProbabilities(values, rng);
    int global_best_arm = argmax_random_ties(arm_probs, workspace, rng);

    size_t ndraws = values.nrow();
    Vector value_remaining(ndraws);
    for (size_t i = 0; i < ndraws; ++i) {
      int best = argmax_random_ties(values.row(i), workspace, rng);
      value_remaining[i] = values(i, best) - values(i, global_best_arm);
    }
    return value_remaining;
  }
}

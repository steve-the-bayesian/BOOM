#ifndef BOOM_STATS_BANDITS_OPTIMAL_ARM_PROBABILITIES_HPP_
#define BOOM_STATS_BANDITS_OPTIMAL_ARM_PROBABILITIES_HPP_
/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "LinAlg/Array.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // Args:
  //   view: The vector whose maximal element is desired.
  //   candidates: A vector of workspace.
  //   rng:  The random number generator used to break ties.
  //
  // Returns:
  //   The index of the largest value in 'view', breaking ties at random.
  size_t argmax_random_ties(const ConstVectorView &view,
                            std::vector<int> &candidates,
                            RNG &rng = GlobalRng::rng);

  // A convenience implementation of the preceding function, potentially
  // slightly slower if called repeatedly.
  inline size_t argmax_random_ties(const ConstVectorView &view,
                            RNG &rng = GlobalRng::rng) {
    std::vector<int> wsp;
    return argmax_random_ties(view, wsp, rng);
  }

  // Compute the optimal arm probabilities for a multi-armed bandit problem.
  //
  // Args:
  //   value: A Matrix containing posterior draws for the expected reward for
  //     each arm of the bandit.  Element (i, j) contains Monte Carlo draw i for
  //     the value of arm j.
  //   rng: The random number generator used to break ties when multiple arms
  //     achieve the maximal value.
  //
  // Returns:
  //   A vector containing the Monte Carlo probability that each arm is the
  //   optimal arm.
  Vector compute_optimal_arm_probabilities(
      const Matrix &value, RNG &rng = GlobalRng::rng);

  // Compute the optimal arm probabilities for a multi-armed bandit problem when
  // the optimal arm probabilities vary across users.
  //
  // Args:
  //   value: A 3-way Array containing posterior draws for the expected reward
  //     for each arm of the bandit.  Element (i, j, a) contains the value for
  //     subject i, Monte Carlo draw j, for the value of arm a.
  //   rng: The random number generator used to break ties when multiple arms
  //     achieve the maximal value.
  //
  // Returns:
  //   A matrix with element (i, j) containing the Monte Carlo probability that
  //   each arm j is the optimal arm for user i.
  Matrix compute_user_specific_optimal_arm_probabilities(
      const Array &value, RNG &rng = GlobalRng::rng);

}  // namespace BOOM


#endif  // BOOM_STATS_BANDITS_OPTIMAL_ARM_PROBABILITIES_HPP_

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

#include "stats/optimal_arm_probabilities.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

#include "cpputil/report_error.hpp"

namespace BOOM {

  size_t argmax_random_ties(const ConstVectorView &view,
                            std::vector<int> &candidates,
                            RNG &rng) {
    if (view.empty()) {
      report_error("Empty view passed to argmax_random_ties.");
    }
    candidates.clear();
    size_t i = 0;
    double max_value = negative_infinity();
    for (double el : view) {
      if (el > max_value) {
        candidates.clear();
        max_value = el;
        candidates.push_back(i);
      } else if (el == max_value) {
        candidates.push_back(i);
      }
      ++i;
    }

    if (candidates.size() == 1) {
      return candidates[0];
    } else {
      uint index = rmulti_mt(rng, 0, candidates.size() - 1);
      return candidates[index];
    }
  }

  Vector compute_optimal_arm_probabilities(const Matrix &values, RNG &rng) {
    Vector probs(values.ncol(), 0.0);
    std::vector<int> candidates;
    for (size_t i = 0; i < values.nrow(); ++i) {
      size_t winner = argmax_random_ties(values.row(i), candidates, rng);
      ++probs[winner];
    }
    probs /= values.nrow();
    return probs;
  }

  Matrix compute_user_specific_optimal_arm_probabilities(
      const Array &values, RNG &rng) {
    if (values.ndim() != 3) {
      report_error("compute_optimal_arm_probabilities needs a 3D array "
                   "as input");
    }
    int num_subjects = values.dim(0);
    int num_iterations = values.dim(1);
    int num_arms = values.dim(2);

    Matrix probs(num_subjects, num_arms);
    std::vector<int> candidates;
    for (int user = 0; user < num_subjects; ++user) {
      for (int i = 0; i < num_iterations; ++i) {
        int winner = argmax_random_ties(values.vector_slice(user, i, -1),
                                        candidates,
                                        rng);
        ++probs(user, winner);
      }
    }
    probs /= num_iterations;
    return probs;
  }

}  // namespace BOOM

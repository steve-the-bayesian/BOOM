/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "numopt/MarkovDecisionProcess.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  Vector MarkovDecisionProcess::value_iteration(
      int horizon, double discount_rate) const {
    Vector old_value(num_states(), 0.0);
    Vector value(num_states());
    for (int i = 0; i < horizon; ++i) {
      for (int r = 0; r < num_states(); ++r) {
        double conditional_value = negative_infinity();
        for (int a = 0; a < num_actions(); ++a) {
          double tmp_value = transition_probabilities_.vector_slice(r, a, -1).dot(
              discount_rate * old_value + rewards_.vector_slice(r, a, -1));
          conditional_value = std::max<double>(conditional_value, tmp_value);
        }
        value(r) = conditional_value;
      }
      old_value = value;
      Vector diff = old_value - value;
      if (diff.max_abs() < 1e-8) {
        return value;
      }
    }
    return value;
  }

  // The optimal policy is derived using value iteration.
  // Args:
  //   horizon:  The number of periods the MDP will run.
  //   discount_rate: A positive number giving the time value of money.  A
  //     discount rate of 1.0 means that one dollar tomorrow has equal value
  //     to one dollar today.  A discount rate of 0.9 means one dollar
  //     tomorrow is worth 90 cents today.
  //
  // Returns:
  //   A vector pi[s] giving the optimal action to take given the current
  //   state s.
  std::vector<int> MarkovDecisionProcess::optimal_policy(
      int horizon, double discount_rate) const {
    Vector value = value_iteration(horizon, discount_rate);
    std::vector<int> policy(num_states());

    for (int s = 0; s < num_states(); ++s) {
      double best_value = negative_infinity();
      int best_action = -1;
      for (int a = 0; a < num_actions(); ++a) {
        double tmp_value = transition_probabilities_.vector_slice(s, a, -1).dot(
            discount_rate * value + rewards_.vector_slice(s, a, -1));
        if (tmp_value > best_value) {
          best_action = a;
          best_value = tmp_value;
        }
      }
      policy[s] = best_action;
    }
    return policy;
  }

  void MarkovDecisionProcess::validate_transition_probabilities(
      const Array &transition_probabilities) {
    if (transition_probabilities.ndim() != 3) {
      report_error("transition_probabilities must be a 3-way array.");
    }

    if (transition_probabilities_.dim(2) != num_states()) {
      report_error("The first and last dimensions of transition_probabilities "
                   "must be the same size.");
    }

    for (int a = 0; a < num_actions(); ++a) {
      for (int r = 0; r < num_states(); ++r) {
        ConstVectorView probs(transition_probabilities.vector_slice(r, a, -1));
        double lo = probs.min();
        double hi = probs.max();
        double total = probs.sum();
        if (lo < 0 || hi > 1.0) {
          report_error("Transition probabilities must all be between 0 and 1.");
        }
        if (fabs(total - 1.0) > 1e-8) {
          report_error("Transition probabilities must sum to 1.");
        }
      }
    }
  }

  void MarkovDecisionProcess::validate_rewards(const Array &rewards) {
    if (rewards.ndim() != 3) {
      report_error("rewards must be a 3-way array.");
    }
    if (rewards.dim(0) != num_states() || rewards.dim(2) != num_states()) {
      report_error("The first and last dimension of rewards must equal the "
                   "number of states.");
    }
    if (rewards.dim(1) != num_actions()) {
      report_error("The middle dimension of rewards must be the number "
                   "of actions.");
    }
  }

}  // namespace BOOM

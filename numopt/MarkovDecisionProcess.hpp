#ifndef BOOM_NUMOPT_DYNPROG_HPP_
#define BOOM_NUMOPT_DYNPROG_HPP_

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

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Array.hpp"

namespace BOOM {

  // A finite state, stationary Markov decsision process.
  class MarkovDecisionProcess {
   public:
    // transition_probabilities: A 3-way array.  Element (r, a, s) is the
    //   probability of transitioning to state s, given that the current state
    //   is r and the current action is a.
    // rewards: Element (r, a, s) is the expected reward under action a when
    //   transitioning from r to s.
    MarkovDecisionProcess(const Array &transition_probabilities,
                          const Array &rewards)
        : transition_probabilities_(transition_probabilities),
          rewards_(rewards)
    {
      validate_transition_probabilities(transition_probabilities);
      validate_rewards(rewards);
    }

    int num_states() const {
      return transition_probabilities_.dim(0);
    }

    int num_actions() const {
      return transition_probabilities_.dim(1);
    }

    // Args:
    //   discount_rate: A positive number giving the time value of money.  A
    //     discount rate of 1.0 means that one dollar tomorrow has equal value
    //     to one dollar today.  A discount rate of 0.9 means one dollar
    //     tomorrow is worth 90 cents today.
    //   horizon:  The number of periods the MDP will run.
    //
    // Returns:
    //   A vector V[s] giving the discounted expected sum of future rewards
    //   given current state s.
    Vector value_iteration(int horizon, double discount_rate) const;

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
    std::vector<int> optimal_policy(int horizon, double discount_rate) const;

   private:
    void validate_transition_probabilities(
        const Array &transition_probabilities);
    void validate_rewards(const Array &rewards);

    Array transition_probabilities_;
    Array rewards_;
  };

}  // namespace BOOMx

#endif  // BOOM_NUMOPT_DYNPROG_HPP_

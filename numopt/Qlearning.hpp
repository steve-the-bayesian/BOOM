#ifndef BOOM_NUMOPT_QLEARNING_HPP_
#define BOOM_NUMOPT_QLEARNING_HPP_

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


#include "LinAlg/Matrix.hpp"

namespace BOOM {

  // Qlearning is a "model free" learning method based purely on states and
  // rewards.
  class Qlearning {
   public:
    // Args:
    //   num_states:  The number of observed state categories.
    //   num_actions:  The number of available actions.
    //   learning_rate: A number between 0 and 1.  Numbers close to zero weight
    //     the past more heavily and are more stable.  Numbers closer to 1
    //     discount the past.
    //   discount_factor: The rate at which future money is discounted to the
    //     present. A discount factor of .9 says that a dollar tomorrow is worth
    //     90 cents today.
    Qlearning(int num_states, int num_actions, double learning_rate,
              double discount_factor);

    // Choose the appropriate action given the current state.
    int choose_action(int state) const;

    // Update the Qtable based on newly observed information.
    // Args:
    //   old_state:  The state the system was in prior to the most recent play.
    //   action:  The action that was chosen based on the old state.
    //   new_state:  The state resulting from the most recent play.
    //   reward:  The reward experienced as a result of the most recent play.
    void update(int old_state, int action, int new_state, double reward);

   private:
    Matrix Qtable_;
    double learning_rate_;
    double discount_factor_;
  };

}  // namespace BOOM


#endif  // BOOM_NUMOPT_QLEARNING_HPP_

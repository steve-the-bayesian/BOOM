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

#include "numopt/Qlearning.hpp"

namespace BOOM {

  Qlearning::Qlearning(int num_states, int num_actions,
                       double learning_rate, double discount_factor)
      : Qtable_(num_states, num_actions),
        learning_rate_(learning_rate),
        discount_factor_(discount_factor)
  {}

  int Qlearning::choose_action(int state) const {
    return Qtable_.row(state).imax();
  }

  void Qlearning::update(int old_state, int action, int new_state, double reward) {
    Qtable_(old_state, action) += learning_rate_ * (
        reward
        + discount_factor_ * Qtable_.row(new_state).max()
        - Qtable_(old_state, action));
  }

}  // namespace BOOM

#ifndef BOOM_HMM_GENERAL_HMM_HPP_
#define BOOM_HMM_GENERAL_HMM_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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
#include "LinAlg/Cholesky.hpp"
#include "Models/DataTypes.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class GeneralContinuousStateHmm
      : virtual public Model
  {
   public:
    GeneralContinuousStateHmm * clone() const override = 0;

    // The dimension of the latent state variable at a single time point.
    virtual int state_dimension() const = 0;

    // Args:
    //   observed_data:  The observed data at time t.
    //   state:  The state vector at time t.
    //   time_index:  The time index at time t.
    //   parameters: A vectorized set of model parameters, suitable for passing
    //     to this->unvectorize_params().
    //
    // Returns:
    //   The log of the observation density at time time_index, evaluated at
    //   observed_data, conditional on state.
    virtual double log_observation_density(const Data &observed_data,
                                           const Vector &state,
                                           int time_index,
                                           const Vector &parameters) const = 0;

    // Evaluate the log of the transition density.
    //
    // Args:
    //   new_state:  The state vector being transitioned to at time t+1.
    //   old_state:  The state vector being transitioned from at time t.
    //   old_time:  The time index at time t.
    //   parameters: A vectorized set of model parameters, suitable for passing
    //     to this->unvectorize_params().
    //
    // Returns:
    //   The log of the transition density.
    virtual double log_transition_density(const Vector &new_state,
                                          const Vector &old_state,
                                          int old_time,
                                          const Vector &parameters) const = 0;

    // Simulate a new value for the state vector at time t+1 given the value of
    // the state vector at time t.
    //
    // Args:
    //   rng:  The random number generator to use for the simulation.
    //   old_state:  The vector of latent state variables at time t.
    //   old_time:  The time index at time t.
    //   parameters: A vectorized set of model parameters, suitable for passing
    //     to this->unvectorize_params().
    //
    // Returns:
    //   A draw from the predictive distribution of state at time t+1 given
    //   state at time t.
    virtual Vector simulate_transition(RNG &rng,
                                       const Vector &old_state,
                                       int old_time,
                                       const Vector &parameters) const = 0;

    // The expected value of the state at time t+1 given parameters and state at
    // time t.
    //
    // Args:
    //   old_state:  The state vector at time t.
    //   old_time:  The time index at time t (which is just t).
    //   parameters: A vectorized set of model parameters, suitable for passing
    //     to this->unvectorize_params().
    virtual Vector predicted_state_mean(const Vector &old_state,
                                        int old_time,
                                        const Vector &parameters) const = 0;
  };



  
}  // namespace BOOM

#endif // BOOM_HMM_GENERAL_HMM_HPP_

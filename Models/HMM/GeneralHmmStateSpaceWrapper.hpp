#ifndef BOOM_GENERAL_HMM_STATE_SPACE_WRAPPER_HPP_
#define BOOM_GENERAL_HMM_STATE_SPACE_WRAPPER_HPP_
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

#include "Models/HMM/GeneralHmm.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/DeferredDataPolicy.hpp"
#include "Models/Policies/NullPriorPolicy.hpp"

namespace BOOM {
  class GeneralHmmStateSpaceWrapper
      : public GeneralContinuousStateHmm,
        public CompositeParamPolicy,
        public DeferredDataPolicy,
        public NullPriorPolicy
  {
   public:
    explicit GeneralHmmStateSpaceWrapper(
        const Ptr<ScalarStateSpaceModelBase> &model);
    
    GeneralHmmStateSpaceWrapper(const GeneralHmmStateSpaceWrapper &rhs);
    GeneralHmmStateSpaceWrapper * clone() const override;
    
    GeneralHmmStateSpaceWrapper & operator=(
        const GeneralHmmStateSpaceWrapper &rhs);
    
    GeneralHmmStateSpaceWrapper(GeneralHmmStateSpaceWrapper &&rhs) = default;
    GeneralHmmStateSpaceWrapper &
    operator=(GeneralHmmStateSpaceWrapper &&rhs) = default;
    
    int state_dimension() const override {
      return model_->state_dimension();
    }

    double log_observation_density(const Data &observed_data,
                                   const Vector &state,
                                   int time_index,
                                   const Vector &parameters) const override;

    double log_transition_density(const Vector &new_state,
                                  const Vector &old_state,
                                  int old_time,
                                  const Vector &parameters) const override;

    Vector simulate_transition(RNG &rng,
                               const Vector &old_state,
                               int old_time,
                               const Vector &parameters) const override {
      ParameterHolder params(model_.get(), parameters);
      return model_->simulate_next_state(rng, old_state, old_time + 1);
    }

    Vector predicted_state_mean(const Vector &old_state,
                                int old_time,
                                const Vector &parameters) const override {
      ParameterHolder params(model_.get(), parameters);
      return *model_->state_transition_matrix(old_time) * old_state;
    }
    
   private:
    mutable Ptr<ScalarStateSpaceModelBase> model_;
  };
}  // namespace BOOM

#endif  // BOOM_GENERAL_HMM_STATE_SPACE_WRAPPER_HPP_

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

#include "Models/StateSpace/Filters/KalmanFilterBase.hpp"
#include "Models/StateSpace/StateSpaceModelBase.hpp"

namespace BOOM {
  namespace Kalman {
    MarginalDistributionBase::MarginalDistributionBase(int dim)
        : state_mean_(dim), state_variance_(dim) {}
  }  // namespace Kalman

  KalmanFilterBase::KalmanFilterBase()
      : status_(NOT_CURRENT), log_likelihood_(negative_infinity()) {}
  
  // If the model adds new parameters after this function is called, then the
  // new parameters will not be observed.  This can happen with a state space
  // model when new components of state are added using add_state().
  //
  // To combat this possibility, the model for the filter should be set as late
  // as possible.
  void KalmanFilterBase::observe_model_parameters(StateSpaceModelBase *model) {
    for (auto &prm : model->parameter_vector()) {
      prm->add_observer([this]() {this->mark_not_current();});
    }
  }

  void KalmanFilterBase::clear() {
    log_likelihood_ = 0;
    status_ = NOT_CURRENT;
  }
  
  
}  // namespace BOOM

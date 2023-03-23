/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StateModels/ObservationParameterManager.hpp"

namespace BOOM {

  ObservationParameterManager::ObservationParameterManager(
      int nseries, int xdim)
  {
    Vector ones(xdim, 1.0);
    Selector include_all(xdim, true);
    for (int i = 0; i < nseries; ++i) {
      coefs_.push_back(new GlmCoefs(ones, include_all));
      suf_.push_back(new WeightedRegSuf(xdim));
    }
  }

  ObservationParameterManager::ObservationParameterManager(
      const ObservationParameterManager &rhs)
  {
    operator=(rhs);
  }

  ObservationParameterManager & ObservationParameterManager::operator=(
      const ObservationParameterManager &rhs) {
    if (&rhs != this) {
      coefs_.clear();
      suf_.clear();
      for (int i = 0; i < rhs.coefs_.size(); ++i) {
        coefs_.push_back(rhs.coefs_[i]->clone());
        suf_.push_back(rhs.suf_[i]->clone());
      }
    }
    return *this;
  }

  Vector ObservationParameterManager::compute_residual(
      const ConstVectorView &state,
      int time,
      const ConditionallyIndependentMultivariateStateSpaceModelBase *host,
      const SharedStateModel *state_model) const {
    // Subtract off the effect of other state models (or regression models), and
    // add in the effect of this one, so that the only effect present is from
    // this state model and random error.
    //
    // The first "state" calculation below uses the full state vector.  The
    // second uses 'now' which is a subset.
    const Selector &observed(host->observed_status(time));
    return host->adjusted_observation(time)
        - (*host->observation_coefficients(time, observed)
           * host->shared_state(time))
        + (*state_model->observation_coefficients(time, observed)) * state;
  }

}  // namespace BOOM

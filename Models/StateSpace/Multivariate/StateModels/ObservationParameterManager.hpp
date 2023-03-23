#ifndef BOOM_STATE_SPACE_MULTIVARIATE_OBSERVATION_PARAMETER_MANAGER_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_OBSERVATION_PARAMETER_MANAGER_HPP_
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

#include "Models/Glm/GlmCoefs.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedStateModel.hpp"

namespace BOOM {
  //===========================================================================
  // A SharedStateModel servicing state space models with a conditionally
  // independent error structure.
  //
  // These models are parameterized by a set of regression coefficients that
  // multiply elements of state.
  class ObservationParameterManager {
   public:
    ObservationParameterManager(int nseries, int xdim);

    ObservationParameterManager(const ObservationParameterManager &rhs);
    ObservationParameterManager & operator=(
        const ObservationParameterManager &rhs);

    ObservationParameterManager(ObservationParameterManager &&rhs) = default;
    ObservationParameterManager & operator=(
        ObservationParameterManager &&rhs) = default;

    int nseries() const {return coefs_.size();}

    // A utility that child classes can use to implement 'observe_state'.
    // Isolate the contributions of this state model from other state models.
    // Then call record_observed_data_given_state, which is implemented at the
    // child level.
    Vector compute_residual(
        const ConstVectorView &state,
        int time,
        const ConditionallyIndependentMultivariateStateSpaceModelBase *host,
        const SharedStateModel *state_model) const;

    Ptr<GlmCoefs> & coefs(int series) {return coefs_[series];}
    const Ptr<GlmCoefs> & coefs(int series) const {return coefs_[series];}

    Ptr<WeightedRegSuf> & suf(int series) {return suf_[series];}
    const Ptr<WeightedRegSuf> & suf(int series) const {return suf_[series];}

    void clear_sufficient_statistics() {
      for (size_t i = 0; i < suf_.size(); ++i) {
        suf_[i]->clear();
      }
    }

   private:
    // The regression coefficients linking the observations to the state.
    // Element i corresponds to series i, with coefs_[i] multiplying some subset
    // of the state.
    std::vector<Ptr<GlmCoefs>> coefs_;

    // Element i contains the sufficient statistics for series i.  The X'X
    // portion of the sufficient statistics will be similar across series.  We
    // need a separate X'X for each series because X'X will only contain entries
    // for which the series was observed, and each series can have its own
    // pattern of missingness.
    //
    // The sufficient statistics are of type WeightedRegSuf instead of just
    // RegSuf in order to accommodate normal mixture error distributions.
    std::vector<Ptr<WeightedRegSuf>> suf_;
  };

}  // namespace BOOM
#endif  //  BOOM_STATE_SPACE_MULTIVARIATE_OBSERVATION_PARAMETER_MANAGER_HPP_

/*
  Copyright (C) 2018-2019 Steven L. Scott

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

#include "Models/StateSpace/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"

namespace BOOM {
  namespace {
    using MVSSMS = MultivariateStateSpaceModelSampler;
  }  // namespace

  MVSSMS::MultivariateStateSpaceModelSampler(
      MultivariateStateSpaceModelBase *model,
      RNG &rng)
      : PosteriorSampler(rng),
        model_(model),
        latent_data_initialized_(false)
  {}

  double MVSSMS::logpri() const {
    double ans = model_->observation_model()->logpri();
    for (int s = 0; s < model_->number_of_state_models(); ++s) {
      ans += model_->state_model(s)->logpri();
    }
    return ans;
  }
  
  void MVSSMS::draw() {
    if (!latent_data_initialized_) {
      model_->impute_state(rng());
      latent_data_initialized_ = true;
      impute_nonstate_latent_data();
    }
    model_->observation_model()->sample_posterior();
    for (int s = 0; s < model_->number_of_state_models(); ++s) {
      model_->state_model(s)->sample_posterior();
    }
    impose_identifiability_constraints();
    // The complete data sufficient statistics for the observation model and the
    // state models are updated when calling impute_state.  The non-state latent
    // data should be imputed immediately before that, so the complete data
    // sufficient statistics reflect all the latent data correctly.
    impute_nonstate_latent_data();
    model_->impute_state(rng());
    // End with a call to impute_state() so that the internal state of
    // the Kalman filter matches up with the parameter draws.
  }

  void MVSSMS::impose_identifiability_constraints() {
    for (int i = 0; i < model_->number_of_state_models(); ++i) {
      model_->state_model(i)->impose_identifiability_constraint();
    }
  }
  
}  // namespace BOOM

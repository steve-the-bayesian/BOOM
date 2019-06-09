/*
  Copyright (C) 2005-2019 Steven L. Scott

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

#include "Models/StateSpace/PosteriorSamplers/MvStateSpaceRegressionPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  namespace {
    using MSSRPS = MultivariateStateSpaceRegressionPosteriorSampler;
    using MSSRM = MultivariateStateSpaceRegressionModel;
  }  // namespace

  MSSRPS::MultivariateStateSpaceRegressionPosteriorSampler(
      MSSRM *model, RNG &rng)
      : PosteriorSampler(rng),
        model_(model),
        latent_data_initialized_(false)
  {
    if (model_->has_series_specific_state()) {
      for (int i = 0; i < model_->nseries(); ++i) {
        Ptr<ProxyScalarStateSpaceModel> proxy =
            model_->series_specific_model(i);
        NEW(StateSpacePosteriorSampler, proxy_sampler)(
            proxy.get());
        proxy->set_method(proxy_sampler);
      }
    }
  }

  void MSSRPS::draw() {
    if (!latent_data_initialized_) {
      // Ensure all state models observe the time dimension, and that space has been
      // allocated for all state structures.
      model_->impute_state(rng());
      latent_data_initialized_ = true;
      impute_nonstate_latent_data();
    } // End latent data initialization.

    // Sample regression parameters and residual variance parameters.
    model_->observation_model()->sample_posterior();
      
    // Sample parameters for the shared state models.
    for (int s = 0; s < model_->number_of_state_models(); ++s) {
      model_->state_model(s)->sample_posterior();
    }

    // Sample parameters for proxy models if any series specific state is
    // present.
    if (model_->has_series_specific_state()) {
      for (int j = 0; j < model_->nseries(); ++j) {
        ProxyScalarStateSpaceModel &proxy(*model_->series_specific_model(j));
        for (int s = 0; s < proxy.number_of_state_models(); ++s) {
          proxy.state_model(s)->sample_posterior();
        }
      }
    }
    
    // The complete data sufficient statistics for the observation model and the
    // state models are updated when calling impute_state.  The non-state latent
    // data should be imputed immediately before that, so the complete data
    // sufficient statistics reflect all the latent data correctly.
    impute_nonstate_latent_data();
    
    // End with a call to impute_state() so that the internal state of
    // the Kalman filter matches up with the parameter draws.
    model_->impute_state(rng());
  }

  double MSSRPS::logpri() const {
    double ans = model_->observation_model()->logpri();
    for (int s = 0; s < model_->number_of_state_models(); ++s) {
      ans += model_->state_model(s)->logpri();
    }
    if (model_->has_series_specific_state()) {
      for (int p = 0; p < model_->nseries(); ++p) {
        const ProxyScalarStateSpaceModel &proxy(
            *model_->series_specific_model(p));
        for (int s = 0; s < proxy.number_of_state_models(); ++s) {
          ans += proxy.state_model(s)->logpri();
        }
      }
    }
    return ans;
  }
    
}  // namespace BOOM

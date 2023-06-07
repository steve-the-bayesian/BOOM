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

#include "Models/StateSpace/Multivariate/PosteriorSamplers/StudentMvssPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  StudentMvssPosteriorSampler::StudentMvssPosteriorSampler(
      StudentMvssRegressionModel *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        latent_data_initialized_(false)
  {}

  StudentMvssPosteriorSampler * StudentMvssPosteriorSampler::clone_to_new_host(
      Model *new_host) const {
    StudentMvssRegressionModel *model = dynamic_cast<StudentMvssRegressionModel *>(
        new_host);
    if (!model) {
      report_error("Wrong type of host passed to "
                   "StudentMvssPosteriorSampler::clone_to_new_host.");
    }
    return new StudentMvssPosteriorSampler(model, rng());
  }

  void StudentMvssPosteriorSampler::draw() {
    if (!latent_data_initialized_) {
      // Ensure all state models observe the time dimension, and that space has been
      // allocated for all state structures.
      model_->impute_state(rng());
      latent_data_initialized_ = true;
    } // End latent data initialization.

    // Sample regression parameters and residual variance parameters.
    model_->observation_model()->sample_posterior();

    // Sample parameters for the shared state models.
    for (int s = 0; s < model_->number_of_state_models(); ++s) {
      model_->state_model(s)->sample_posterior();
    }

    // Sample parameters for proxy models if any series specific state is
    // present.
    using Proxy = ProxyScalarStateSpaceModel<StudentMvssRegressionModel>;
    if (model_->has_series_specific_state()) {
      for (int j = 0; j < model_->nseries(); ++j) {
        Proxy &proxy(*model_->series_specific_model(j));
        for (int s = 0; s < proxy.number_of_state_models(); ++s) {
          proxy.state_model(s)->sample_posterior();
        }
      }
    }

    // End with a call to impute_state() so that the internal state of
    // the Kalman filter matches up with the parameter draws.
    model_->impute_state(rng());
  }


}  // namespace BOOM

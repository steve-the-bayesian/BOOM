// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_STATE_SPACE_POISSON_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_POISSON_POSTERIOR_SAMPLER_HPP_

#include "Models/Glm/PosteriorSamplers/PoissonDataImputer.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionSpikeSlabSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateSpacePoissonModel.hpp"

namespace BOOM {
  class StateSpacePoissonPosteriorSampler : public StateSpacePosteriorSampler {
   public:
    // Args:
    //   model: The model for which posterior samples are desired.
    //     All state components should have posterior samplers
    //     assigned to them before 'model' is passed to this
    //     constructor.  Likewise, the observation model should have
    //     observation_model_sampler assigned to it before being
    //     passed here.
    //   observation_model_sampler: The posterior sampler for the
    //     Poisson regression observation model.  We need a separate
    //     handle to this sampler because we need to control the
    //     latent data imputation and parameter draw steps separately.
    StateSpacePoissonPosteriorSampler(
        StateSpacePoissonModel *model,
        const Ptr<PoissonRegressionSpikeSlabSampler> &observation_model_sampler,
        RNG &seeding_rng = GlobalRng::rng);

    StateSpacePoissonPosteriorSampler *clone_to_new_host(
        Model *new_host) const override;

    // Impute the latent Gaussian observations and variances at each
    // data point.
    void impute_nonstate_latent_data() override;

    // Clear the complete_data_sufficient_statistics for the Poisson
    // regression model.
    void clear_complete_data_sufficient_statistics();

    // Increment the complete_data_sufficient_statistics for the
    // Poisson regression model by adding the latent data from
    // observation t.  This update is conditional on the contribution
    // of the state space portion of the model, which is stored in the
    // "offset" component of observation t.
    void update_complete_data_sufficient_statistics(int t);

   private:
    StateSpacePoissonModel *model_;
    Ptr<PoissonRegressionSpikeSlabSampler> observation_model_sampler_;
    PoissonDataImputer data_imputer_;
  };
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_POISSON_POSTERIOR_SAMPLER_HPP_

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

#ifndef BOOM_SRC_MODELS_STATE_SPACE_STUDENT_POSTERIOR_SAMPLER_HPP_
#define BOOM_SRC_MODELS_STATE_SPACE_STUDENT_POSTERIOR_SAMPLER_HPP_

#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateSpaceStudentRegressionModel.hpp"

namespace BOOM {

  class StateSpaceStudentPosteriorSampler : public StateSpacePosteriorSampler {
   public:
    StateSpaceStudentPosteriorSampler(
        StateSpaceStudentRegressionModel *model,
        const Ptr<TRegressionSpikeSlabSampler> &observation_model_sampler,
        RNG &seeding_rng = GlobalRng::rng);

    // Impute the latent variances at each data point.
    void impute_nonstate_latent_data() override;

    // Clear the complete_data_sufficient_statistics for the weighted
    // regression model.
    void clear_complete_data_sufficient_statistics();

    // Increment the complete_data_sufficient_statistics for the
    // weighted regression model by adding the data from observation t
    // along with its imputed variance.  This update is conditional on
    // the contribution of the state space portion of the model, which
    // is stored in the "offset" component of observation t.
    void update_complete_data_sufficient_statistics(int t);

   private:
    StateSpaceStudentRegressionModel *model_;
    Ptr<TRegressionSpikeSlabSampler> observation_model_sampler_;
    TDataImputer data_imputer_;
  };

}  // namespace BOOM

#endif  //  BOOM_SRC_MODELS_STATE_SPACE_STUDENT_POSTERIOR_SAMPLER_HPP_

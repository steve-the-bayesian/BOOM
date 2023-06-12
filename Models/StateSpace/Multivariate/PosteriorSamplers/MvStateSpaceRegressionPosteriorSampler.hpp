#ifndef BOOM_STATE_SPACE_MULTIVARIATE_REGRESSION_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_REGRESSION_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2019 Steven L. Scott

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

// The name of this file is abbreviated to stay below the 100 character limit
// in ancient, archaic tar formats.
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class MultivariateStateSpaceRegressionPosteriorSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model: The model to be managed.  Calling this constructor sets a
    //     StateSpacePosteriorSampler for each of the proxy models managing
    //     series-specific state.  It is assumed that posterior samplers will be
    //     set directly for
    //     - model->observation_model()
    //     - All shared-state models owned by model.
    //     - All state models owned by each proxy model.
    //   seeding_rng: The random number generator used to set the seed
    //     for the RNG owned by this sampler.
    explicit MultivariateStateSpaceRegressionPosteriorSampler(
        MultivariateStateSpaceRegressionModel *model,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

   private:
    MultivariateStateSpaceRegressionModel *model_;
    bool latent_data_initialized_;

    // A stub for when non-gaussian data becomes supported.
    virtual void impute_nonstate_latent_data() {}
  };

}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_MULTIVARIATE_REGRESSION_POSTERIOR_SAMPLER_HPP_

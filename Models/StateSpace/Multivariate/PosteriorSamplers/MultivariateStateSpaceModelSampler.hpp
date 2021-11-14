#ifndef BOOM_STATE_SPACE_MULTIVARIATE_STATE_SPACE_MODEL_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_MULTIVARIATE_STATE_SPACE_MODEL_POSTERIOR_SAMPLER_HPP_

/*
  Copyright (C) 2018 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceModelBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class MultivariateStateSpaceModelSampler
      : public PosteriorSampler {
   public:
    explicit MultivariateStateSpaceModelSampler(
        MultivariateStateSpaceModelBase *model,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    // If the state models or error distribution are mixtures of normals, then
    // unmix them.
    virtual void impute_nonstate_latent_data() {}

    void impose_identifiability_constraints();

   private:
    MultivariateStateSpaceModelBase *model_;
    bool latent_data_initialized_;
  };

}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_MULTIVARIATE_STATE_SPACE_MODEL_POSTERIOR_SAMPLER_HPP_

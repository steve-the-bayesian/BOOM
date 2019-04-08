#ifndef BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_POSTERIOR_SAMPLER_HPP_
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

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/GammaModel.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"

namespace BOOM {

  class SharedLocalLevelPosteriorSampler
      : public PosteriorSampler {
   public:
    // A posterior sampler for the shared local level model.  The prior is a
    // product of independent inverse gamma distributions on the innovation
    // variances, with an independent matrix normal prior on the observation
    // coefficients.  If there are k "local level" factors alpha_1, ..., alpha_k
    // obeying alpha_{jt+1} = alpha_{jt} + eta_{jt}, where eta_{jt} ~ N(0,
    // tausq_{j}) then 1/tausq_j ~ Gamma(a, b), independent of everything else.
    //
    // The prior for the coefficients is matrix normal.  If y[t] = Z * alpha[t]
    // + epsilon[t] ~ N(0, Sigma[t])
    //
    // Args:
    //   model: The shared local level state model to be sampled.
    //   innovation_precision_priors: Independent prior distributions for the
    //     precisions of the random walk innovations.  One prior is needed for
    //     each random factor in 'model.'

    //   coefficient_prior_mean: A 'ydim x nfactors' matrix containing the prior
    //     mean of the observation coefficients.

    //   
    
    //   seeding_rng: The random number generator used to seed the RNG for this
    //     sampler.
    SharedLocalLevelPosteriorSampler(
        SharedLocalLevelStateModel *model,
        const std::vector<Ptr<GammaModelBase>> &innovation_precision_priors,
        const Matrix &coefficient_prior_mean,
        double coefficient_prior_sample_size,
        RNG &seeding_rng = GlobalRng::rng);
                                     
    void draw() override;
    double logpri() const override;
    
   private:
    SharedLocalLevelStateModel *model_;
    std::vector<Ptr<GammaModelBase>> innovation_precision_priors_;
    std::vector<GenericGaussianVarianceSampler> variance_samplers_;
    Ptr<MultivariateRegressionSampler> observation_coefficient_sampler_;
  };
  
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_POSTERIOR_SAMPLER_HPP_


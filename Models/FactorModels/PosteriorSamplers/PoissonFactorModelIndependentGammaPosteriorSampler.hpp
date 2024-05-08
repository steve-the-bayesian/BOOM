#ifndef POISSON_FACTOR_MODEL_INDEPENDENT_GAMMA_POSTERIOR_SAMPLER_HPP
#define POISSON_FACTOR_MODEL_INDEPENDENT_GAMMA_POSTERIOR_SAMPLER_HPP
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

#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/FactorModels/PosteriorSamplers/PoissonFactorPosteriorSamplerBase.hpp"
#include "Models/GammaModel.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  // A PosteriorSampler for the PoissonFactorModel under a conditionally
  // conjugate prior, where the prior for each visitor's demographic category is
  // an independent discrete distribution, and the prior for each site's
  // intensity parametes is the product of independent gamma distributions.
  class PoissonFactorModelIndependentGammaPosteriorSampler
      : public PoissonFactorPosteriorSamplerBase
  {
   public:
    // Args:
    //   model:  The model to be posterior sampled.
    //   default_prior_class_probabilities: The default prior to use
    //     for visitors whose membership probabilities are not otherwise
    //     declared.
    //   default_intensity_prior: Prior distribution to use for a site's
    //     intensity parameters when not otherwise declared.
    //   seeding_rng: The random number generator used to seed this object's
    //     RNG.
    PoissonFactorModelIndependentGammaPosteriorSampler(
        PoissonFactorModel *model,
        const Vector &default_prior_class_probabilities,
        const std::vector<Ptr<GammaModelBase>> &default_intensity_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;

    double logpri() const override {
      // Just to get things compiled.
      return negative_infinity();
    }

    void draw_site_parameters();

    void set_intensity_prior(
        const std::string &site_id,
        const std::vector<Ptr<GammaModelBase>> &prior);

    const std::vector<Ptr<GammaModelBase>> &intensity_prior(
        const std::string &site_id) const;

   private:
    std::vector<Ptr<GammaModelBase>> default_intensity_prior_;
    std::map<std::string, std::vector<Ptr<GammaModelBase>>>
    intensity_parameter_priors_;

    Int iteration_;
  };

}  // namespace BOOM


#endif // POISSON_FACTOR_MODEL_INDEPENDENT_GAMMA_POSTERIOR_SAMPLER_HPP

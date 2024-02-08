#ifndef BOOM_POISSON_FACTOR_HIERARCHICAL_SAMPLER_HPP_
#define BOOM_POISSON_FACTOR_HIERARCHICAL_SAMPLER_HPP_

/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "Models/FactorModels/PosteriorSamplers/PoissonFactorPosteriorSamplerBase.hpp"
#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/MvnModel.hpp"

namespace BOOM {

  // A PosteriorSampler for a PoissonFactorModel under a hierarchical prior that
  // shrinks a site's user profile towards a global mean.

  // Each user's latent demographic class  is assumed
  class PoissonFactorHierarchialSampler
      : public PoissonFactorPosteriorSamplerBase {
   public:
    PoissonFactorHierarchialSampler(
        PoissonFactorModel *model,
        const Vector &default_prior_class_probabilities,
        const Ptr<MvnModel> &profile_hyperprior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;

    // The lambda site parameters are decomponsed into lambda = alpha * pi,
    // where alpha is a scalar and pi is a discrete probability distribution.
    void draw_site_parameters();
    void draw_hyperparameters();

    // Draw the sum of the intensity parameters for a given site.
    void draw_site_parameters_MH(PoissonFactor::Site *site);
    void draw_site_parameter_total(PoissonFactor::Site *site);
    void draw_site_parameter_profile(PoissonFactor::Site *site, double total);

   private:
    void check_dimension(const Ptr<MvnModel> &profile_hyperprior) const;

    PoissonFactorModel *model_;
    Ptr<MvnModel> profile_hyperprior_;
  };

}  // namespace BOOM


#endif  //  BOOM_POISSON_FACTOR_HIERARCHICAL_SAMPLER_HPP_

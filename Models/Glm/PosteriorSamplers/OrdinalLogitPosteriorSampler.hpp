#ifndef BOOM_ORDINAL_LOGIT_POSTEIROR_SAMPLER_HPP_
#define BOOM_ORDINAL_LOGIT_POSTEIROR_SAMPLER_HPP_
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

#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

#include "Models/Glm/OrdinalCutpointModel.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"

#include "Models/Glm/PosteriorSamplers/OrdinalLogitImputer.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/NormalMixtureApproximation.hpp"

#include "Models/MvnBase.hpp"
#include "Models/VectorModel.hpp"

#include "Samplers/UnivariateSliceSampler.hpp"

#include "distributions/rng.hpp"

namespace BOOM {

  class OrdinalLogitPosteriorSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model: The model to be managed.
    //   coefficient_prior: Prior distribution on the coefficients of the model.
    //   cutpoint_prior: Prior distribution on the vector of cutpoints.  Only
    //     positive, finite cutpoints are considered.  The prior should assign
    //     weight zero to illegal cutpoint configurations.
    //   seeding_rng: The random number generator used to seed the RNG for this
    //     sampler.
    OrdinalLogitPosteriorSampler(OrdinalLogitModel *model,
                                 const Ptr<MvnBase> &coefficient_prior,
                                 const Ptr<VectorModel> &cutpoint_prior,
                                 RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

   private:
    void impute_latent_data();
    void draw_beta();
    void draw_cutpoints();

    OrdinalLogitModel *model_;
    Ptr<MvnBase> coefficient_prior_;
    Ptr<VectorModel> cutpoint_prior_;

    WeightedRegSuf complete_data_suf_;
    OrdinalLogitImputer imputer_;
    LogitMixtureApproximation logit_mixture_;

    SpikeSlabSampler coefficient_sampler_;

    // Separate cutpoint samplers are maintained so that the upper and lower
    // limits for each can be manipulated.
    std::vector<ScalarSliceSampler> cutpoint_samplers_;
  };
  
}  // namespace BOOM

#endif  //  BOOM_ORDINAL_LOGIT_POSTEIROR_SAMPLER_HPP_


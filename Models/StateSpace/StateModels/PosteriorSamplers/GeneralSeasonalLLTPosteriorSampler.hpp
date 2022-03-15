#ifndef BOOM_SEASONAL_STATE_MODEL_LLT_HPP_
#define BOOM_SEASONAL_STATE_MODEL_LLT_HPP_
/*
  Copyright (C) 2005-2021 Steven L. Scott

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

#include "Models/StateSpace/StateModels/GeneralSeasonalStateModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnConjSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"

namespace BOOM {

  //===========================================================================
  class GeneralSeasonalLLTPosteriorSampler
      : public PosteriorSampler
  {
   public:
    GeneralSeasonalLLTPosteriorSampler(GeneralSeasonalLLT *model,
                                       const std::vector<Ptr<WishartModel>> &priors,
                                       RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

   private:
    GeneralSeasonalLLT *model_;
    std::vector<Ptr<WishartModel>> priors_;
    std::vector<Ptr<ZeroMeanMvnConjSampler>> subordinate_samplers_;
  };

  //===========================================================================
  class GeneralSeasonalLLTIndependenceSampler
      : public PosteriorSampler
  {
   public:
    GeneralSeasonalLLTIndependenceSampler(
      GeneralSeasonalLLT *model,
      const std::vector<Ptr<GammaModelBase>> &level_innovation_priors,
      const std::vector<Ptr<GammaModelBase>> &slope_innovation_priors,
      RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void set_level_sigma_max(int season, double sigma_max) {
      level_innovation_samplers_[season]->set_sigma_upper_limit(sigma_max);
    }

    void set_slope_sigma_max(int season, double sigma_max) {
      slope_innovation_samplers_[season]->set_sigma_upper_limit(sigma_max);
    }

   private:
    GeneralSeasonalLLT *model_;
    std::vector<Ptr<GammaModelBase>> level_innovation_priors_;
    std::vector<Ptr<GammaModelBase>> slope_innovation_priors_;

    std::vector<Ptr<ZeroMeanMvnIndependenceSampler>> level_innovation_samplers_;
    std::vector<Ptr<ZeroMeanMvnIndependenceSampler>> slope_innovation_samplers_;
  };

}  // namespace BOOM

#endif  // BOOM_SEASONAL_STATE_MODEL_LLT_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#ifndef BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_SAMPLER_HPP_
#define BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_SAMPLER_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/Hierarchical/HierarchicalZeroInflatedPoissonModel.hpp"
#include "Models/PosteriorSamplers/BetaPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/GammaPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/ZeroInflatedPoissonSampler.hpp"
#include "Samplers/ScalarSliceSampler.hpp"

namespace BOOM {

  class HierarchicalZeroInflatedPoissonSampler : public PosteriorSampler {
   public:
    HierarchicalZeroInflatedPoissonSampler(
        HierarchicalZeroInflatedPoissonModel *model,
        const Ptr<DoubleModel> &lambda_mean_prior,
        const Ptr<DoubleModel> &lambda_sample_size_prior,
        const Ptr<DoubleModel> &zero_probability_mean_prior,
        const Ptr<DoubleModel> &zero_probability_sample_size_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

   private:
    HierarchicalZeroInflatedPoissonModel *model_;
    Ptr<DoubleModel> lambda_mean_prior_;
    Ptr<DoubleModel> lambda_sample_size_prior_;
    Ptr<DoubleModel> zero_probability_mean_prior_;
    Ptr<DoubleModel> zero_probability_sample_size_prior_;

    GammaPosteriorSamplerBeta lambda_prior_sampler_;
    BetaPosteriorSampler zero_probability_prior_sampler_;
  };

}  // namespace BOOM

#endif  // BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_SAMPLER_HPP_

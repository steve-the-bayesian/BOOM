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

#ifndef BOOM_MODELS_POSTERIOR_SAMPLERS_GAMMA_POSTERIOR_SAMPLER_HPP_
#define BOOM_MODELS_POSTERIOR_SAMPLERS_GAMMA_POSTERIOR_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Samplers/ScalarSliceSampler.hpp"

namespace BOOM {

  // The GammaPosteriorSampler assumes independent priors on a/b and
  // a.
  class GammaPosteriorSampler : public PosteriorSampler {
   public:
    GammaPosteriorSampler(GammaModel *model,
                          const Ptr<DoubleModel> &mean_prior,
                          const Ptr<DoubleModel> &alpha_prior,
                          RNG &seeding_rng = GlobalRng::rng);

    GammaPosteriorSampler *clone_to_new_host(Model *new_host) const override;

    void draw() override;
    double logpri() const override;

   private:
    GammaModel *model_;
    Ptr<DoubleModel> mean_prior_;
    Ptr<DoubleModel> alpha_prior_;
    ScalarSliceSampler mean_sampler_;
    ScalarSliceSampler alpha_sampler_;
  };

  // GammaPosteriorSamplerBeta assumes independent priors on a/b and
  // b.  It is otherwise identical to GammaPosteriorSampler.
  class GammaPosteriorSamplerBeta : public PosteriorSampler {
   public:
    GammaPosteriorSamplerBeta(GammaModel *model,
                              const Ptr<DoubleModel> &mean_prior,
                              const Ptr<DoubleModel> &beta_prior,
                              RNG &seeding_rng = GlobalRng::rng);
    GammaPosteriorSamplerBeta *clone_to_new_host(Model *model) const override;
    void draw() override;
    double logpri() const override;

   private:
    GammaModel *model_;
    Ptr<DoubleModel> mean_prior_;
    Ptr<DoubleModel> beta_prior_;
    ScalarSliceSampler mean_sampler_;
    ScalarSliceSampler beta_sampler_;
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_POSTERIOR_SAMPLERS_GAMMA_POSTERIOR_SAMPLER_HPP_

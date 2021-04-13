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

#ifndef BOOM_FINITE_MIXTURE_POSTERIOR_SAMPLER_HPP_
#define BOOM_FINITE_MIXTURE_POSTERIOR_SAMPLER_HPP_

#include "Models/FiniteMixtureModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class FiniteMixturePosteriorSampler : public PosteriorSampler {
   public:
    explicit FiniteMixturePosteriorSampler(FiniteMixtureModel *model,
                                           RNG &seeding_rng = GlobalRng::rng)
        : PosteriorSampler(seeding_rng), model_(model) {}

    FiniteMixturePosteriorSampler *clone_to_new_host(
        Model *new_host) const override {
      return new FiniteMixturePosteriorSampler(
          dynamic_cast<FiniteMixtureModel *>(new_host),
          rng());
    }

    double logpri() const override {
      double ans = model_->mixing_distribution()->logpri();
      int S = model_->number_of_mixture_components();
      for (int s = 0; s < S; ++s) {
        ans += model_->mixture_component(s)->logpri();
      }
      return ans;
    }

    void draw() override {
      model_->impute_latent_data(rng());
      model_->mixing_distribution()->sample_posterior();
      for (int s = 0; s < model_->number_of_mixture_components(); ++s) {
        model_->mixture_component(s)->sample_posterior();
      }
    }

   private:
    FiniteMixtureModel *model_;
  };
}  // namespace BOOM
#endif  //  BOOM_FINITE_MIXTURE_POSTERIOR_SAMPLER_HPP_

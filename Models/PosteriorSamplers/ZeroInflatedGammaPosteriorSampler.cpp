// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/PosteriorSamplers/ZeroInflatedGammaPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  ZeroInflatedGammaPosteriorSampler::ZeroInflatedGammaPosteriorSampler(
      ZeroInflatedGammaModel *model,
      const Ptr<BetaModel> &prior_for_nonzero_probability,
      const Ptr<DoubleModel> &prior_for_gamma_mean,
      const Ptr<DoubleModel> &prior_for_gamma_shape, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        prior_for_nonzero_probability_(prior_for_nonzero_probability),
        prior_for_gamma_mean_(prior_for_gamma_mean),
        prior_for_gamma_shape_(prior_for_gamma_shape),
        binomial_sampler_(new BetaBinomialSampler(model->Binomial_model().get(),
                                                  prior_for_nonzero_probability,
                                                  seeding_rng)),
        gamma_sampler_(new GammaPosteriorSampler(
            model->Gamma_model().get(), prior_for_gamma_mean,
            prior_for_gamma_shape, seeding_rng)) {}

  ZeroInflatedGammaPosteriorSampler*
  ZeroInflatedGammaPosteriorSampler::clone_to_new_host(Model *new_host) const {
    return new ZeroInflatedGammaPosteriorSampler(
        dynamic_cast<ZeroInflatedGammaModel *>(new_host),
        prior_for_nonzero_probability_->clone(),
        prior_for_gamma_mean_->clone(),
        prior_for_gamma_shape_->clone(),
        rng());
  }

  double ZeroInflatedGammaPosteriorSampler::logpri() const {
    return binomial_sampler_->logpri() + gamma_sampler_->logpri();
  }

  void ZeroInflatedGammaPosteriorSampler::draw() {
    binomial_sampler_->draw();
    gamma_sampler_->draw();
  }

}  // namespace BOOM

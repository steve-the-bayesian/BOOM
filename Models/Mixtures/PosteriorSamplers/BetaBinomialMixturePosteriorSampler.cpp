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

#include "Models/Mixtures/PosteriorSamplers/BetaBinomialMixturePosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  BetaBinomialMixturePosteriorSampler::BetaBinomialMixturePosteriorSampler(
      BetaBinomialMixtureModel *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model)
  {}

  double BetaBinomialMixturePosteriorSampler::logpri() const {
    double ans = model_->mixing_distribution()->logpri();
    for (int s = 0; s < model_->number_of_mixture_components(); ++s) {
      ans += model_->mixture_component(s)->logpri();
    }
    return ans;
  }

  void BetaBinomialMixturePosteriorSampler::draw() {
    model_->impute_latent_data(rng());
    model_->mixing_distribution()->sample_posterior();
    for (int s = 0; s < model_->number_of_mixture_components(); ++s) {
      model_->mixture_component(s)->sample_posterior();
    }
  }

}  // namespace BOOM

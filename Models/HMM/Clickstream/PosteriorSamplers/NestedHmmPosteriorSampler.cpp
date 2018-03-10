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

#include "Models/HMM/Clickstream/PosteriorSamplers/NestedHmmPosteriorSampler.hpp"

namespace BOOM {

  NestedHmmPosteriorSampler::NestedHmmPosteriorSampler(NestedHmm *model,
                                                       RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), first_time_(true) {}

  double NestedHmmPosteriorSampler::logpri() const {
    double ans = model_->session_model()->logpri();
    for (int H = 0; H < model_->S2(); ++H) {
      ans += model_->event_model(H)->logpri();
      for (int h = 0; h < model_->S1(); ++h) {
        ans += model_->mix(H, h)->logpri();
      }
    }
    return ans;
  }

  void NestedHmmPosteriorSampler::draw() {
    if (first_time_) {
      model_->impute_latent_data();
    }
    model_->session_model()->sample_posterior();
    for (int H = 0; H < model_->S2(); ++H) {
      model_->event_model(H)->sample_posterior();
      for (int h = 0; h < model_->S1(); ++h) {
        model_->mix(H, h)->sample_posterior();
      }
    }
  }

}  // namespace BOOM

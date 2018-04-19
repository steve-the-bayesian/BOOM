// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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

#include "Models/HMM/PosteriorSamplers/HmmPosteriorSampler.hpp"
#include <future>
#include "Models/HMM/HmmFilter.hpp"

namespace BOOM {

  HmmPosteriorSampler::HmmPosteriorSampler(HiddenMarkovModel *hmm,
                                           RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        hmm_(hmm),
        use_threads_(false),
        thread_pool_(0),
        first_time_(true)
  {}

  void HmmPosteriorSampler::draw() {
    if (first_time_) {
      hmm_->impute_latent_data();
      first_time_ = false;
    }
    hmm_->mark()->sample_posterior();
    draw_mixture_components();
    // by drawing latent data at the end, the log likelihood stored
    // int the model matches the current set of parameters.
    hmm_->impute_latent_data();
  }

  double HmmPosteriorSampler::logpri() const {
    double ans = hmm_->mark()->logpri();
    std::vector<Ptr<MixtureComponent>> mix = hmm_->mixture_components();
    uint S = mix.size();
    for (uint s = 0; s < S; ++s) ans += mix[s]->logpri();
    return ans;
  }

  void HmmPosteriorSampler::draw_mixture_components() {
    std::vector<Ptr<MixtureComponent>> mix = hmm_->mixture_components();
    uint S = mix.size();

    if (use_threads_) {
      if (workers_.size() != S) {
        use_threads(true);
      }
      std::vector<std::future<void>> futures;
      for (uint s = 0; s < S; ++s) {
        futures.emplace_back(thread_pool_.submit(workers_[s]));
      }
      for (uint s = 0; s < S; ++s) {
        futures[s].get();
      }
    } else {
      for (uint s = 0; s < S; ++s) {
        mix[s]->sample_posterior();
      }
    }
  }

  void HmmPosteriorSampler::use_threads(bool yn) {
    use_threads_ = yn;
    if (!use_threads_) {
      thread_pool_.set_number_of_threads(0);
      workers_.clear();
    } else {
      std::vector<Ptr<MixtureComponent>> mix = hmm_->mixture_components();
      uint S = mix.size();
      workers_.clear();
      for (uint s = 0; s < S; ++s) {
        workers_.emplace_back(mix[s].get());
      }
    }
  }

}  // namespace BOOM

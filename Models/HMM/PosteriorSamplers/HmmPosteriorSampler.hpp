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

#ifndef BOOM_HMM_POSTERIOR_SAMPLER_HPP
#define BOOM_HMM_POSTERIOR_SAMPLER_HPP

#include "Models/HMM/HMM2.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/ThreadTools.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class MixtureComponentSampler {
   public:
    explicit MixtureComponentSampler(Model *m) : m_(m) {}
    void operator()() { m_->sample_posterior(); }
    // m_ must have been assigned a thread safe PosteriorSampler, or this
    // will result in a race condition on the random seed of the RNG
   private:
    Model *m_;
  };

  class HmmPosteriorSampler : public PosteriorSampler {
   public:
    explicit HmmPosteriorSampler(HiddenMarkovModel *hmm,
                                 RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;
    void use_threads(bool yn = true);
    void draw_mixture_components();

   private:
    HiddenMarkovModel *hmm_;
    std::vector<MixtureComponentSampler> workers_;
    bool use_threads_;
    ThreadWorkerPool thread_pool_;
    // 
    bool first_time_;
  };

}  // namespace BOOM
#endif  // BOOM_HMM_POSTERIOR_SAMPLER_HPP

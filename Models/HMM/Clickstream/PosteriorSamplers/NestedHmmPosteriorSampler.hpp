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

#ifndef BOOM_NESTED_HMM_POSTERIOR_SAMPLER_HPP_
#define BOOM_NESTED_HMM_POSTERIOR_SAMPLER_HPP_

#include "Models/HMM/Clickstream/NestedHmm.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // To use this sampler, make sure all the mixture components and
  // latent Markov models have been assigned posterior samplers.  This
  // class will simply call them all.
  class NestedHmmPosteriorSampler : public PosteriorSampler {
   public:
    explicit NestedHmmPosteriorSampler(NestedHmm *model,
                                       RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;

   private:
    NestedHmm *model_;
    bool first_time_;
  };

}  // namespace BOOM

#endif  //  BOOM_NESTED_HMM_POSTERIOR_SAMPLER_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#ifndef BOOM_BINOMIAL_PROBIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_BINOMIAL_PROBIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_

#include "Models/Glm/BinomialProbitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialProbitSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialProbitTimSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class BinomialProbitCompositeSpikeSlabSampler : public PosteriorSampler {
   public:
    BinomialProbitCompositeSpikeSlabSampler(
        BinomialProbitModel *model, const Ptr<MvnBase> &slab,
        const Ptr<VariableSelectionPrior> &spike, int clt_threshold = 10,
        double proposal_df = 3, RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;

    // If tf == true then draw_model_indicators is a no-op.  Otherwise
    // model indicators will be sampled each iteration.
    void allow_model_selection(bool tf) {
      spike_slab_sampler_.allow_model_selection(tf);
    }

    // In very large problems you may not want to sample every element
    // of the inclusion vector each time.  If max_flips is set to a
    // positive number then at most that many randomly chosen
    // inclusion indicators will be sampled.
    void limit_model_selection(int max_flips) {
      spike_slab_sampler_.limit_model_selection(max_flips);
    }

    // Args:
    //   weights: A vector of length 2 giving the probabilities of
    //     drawing from each sub-sampler.  The first element is the
    //     probability of drawing from the spike and slab sampler.
    //     The second is the probability of drawing from the TIM
    //     sampler.  TIM tends to mix faster, but does not change
    //     which variables are in and out of the model.  spike and
    //     slab is slower, but can alter model dimension.
    void set_sampling_weights(const Vector &weights);

   private:
    BinomialProbitModel *model_;
    Ptr<MvnBase> slab_;
    Ptr<VariableSelectionPrior> spike_;
    BinomialProbitSpikeSlabSampler spike_slab_sampler_;
    BinomialProbitTimSampler tim_;
    Vector sampling_weights_;
  };
}  // namespace BOOM

#endif  // BOOM_BINOMIAL_PROBIT_COMPOSITE_SPIKE_SLAB_SAMPLER_HPP_

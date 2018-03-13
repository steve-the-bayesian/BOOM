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

#ifndef BOOM_BINOMIAL_PROBIT_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_BINOMIAL_PROBIT_SPIKE_SLAB_SAMPLER_HPP_

#include "Models/Glm/BinomialProbitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialProbitDataImputer.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "distributions/rng.hpp"

namespace BOOM {

  class BinomialProbitSpikeSlabSampler : public PosteriorSampler {
   public:
    BinomialProbitSpikeSlabSampler(
        BinomialProbitModel *model, const Ptr<MvnBase> &slab_prior,
        const Ptr<VariableSelectionPrior> &spike_prior, int clt_threshold = 10,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    // If tf == true then draw_model_indicators is a no-op.  Otherwise
    // model indicators will be sampled each iteration.
    void allow_model_selection(bool tf);

    // In very large problems you may not want to sample every element
    // of the inclusion vector each time.  If max_flips is set to a
    // positive number then at most that many randomly chosen
    // inclusion indicators will be sampled.
    void limit_model_selection(int max_flips);

    void impute_latent_data();

    void refresh_xtx();
    WeightedRegSuf complete_data_sufficient_statistics() const;

   private:
    BinomialProbitModel *model_;
    Ptr<MvnBase> slab_prior_;
    Ptr<VariableSelectionPrior> spike_prior_;
    SpikeSlabSampler spike_slab_;
    BinomialProbitDataImputer imputer_;

    SpdMatrix xtx_;
    Vector xtz_;
  };

}  // namespace BOOM

#endif  //  BOOM_BINOMIAL_PROBIT_SPIKE_SLAB_SAMPLER_HPP_

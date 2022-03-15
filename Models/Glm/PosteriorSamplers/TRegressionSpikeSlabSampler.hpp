// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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
#ifndef BOOM_TREGRESSION_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_TREGRESSION_SPIKE_SLAB_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/TRegressionSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/MvnBase.hpp"

namespace BOOM {

  class TRegressionSpikeSlabSampler : public TRegressionSampler {
   public:
    TRegressionSpikeSlabSampler(
        TRegressionModel *model,
        const Ptr<MvnBase> &coefficient_slab_prior,
        const Ptr<VariableSelectionPrior> &coefficient_spike_prior,
        const Ptr<GammaModelBase> &siginv_prior,
        const Ptr<DoubleModel> &nu_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void draw_model_indicators();
    void draw_included_coefficients();
    void allow_model_selection(bool allow);
    void limit_model_selection(int max_flips);

   private:
    TRegressionModel *model_;
    SpikeSlabSampler sam_;
    Ptr<MvnBase> coefficient_slab_prior_;
    Ptr<VariableSelectionPrior> coefficient_spike_prior_;
    Ptr<GammaModelBase> siginv_prior_;
    Ptr<DoubleModel> nu_prior_;
  };

}  // namespace BOOM

#endif  //  BOOM_TREGRESSION_SPIKE_SLAB_SAMPLER_HPP_

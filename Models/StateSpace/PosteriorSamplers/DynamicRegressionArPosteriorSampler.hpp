// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_DYNAMIC_REGRESSION_AUTOREGRESSIVE_POSTERIOR_SAMPLER_HPP_
#define BOOM_DYNAMIC_REGRESSION_AUTOREGRESSIVE_POSTERIOR_SAMPLER_HPP_

#include "Models/StateSpace/StateModels/DynamicRegressionArStateModel.hpp"
#include "Models/TimeSeries/PosteriorSamplers/ArPosteriorSampler.hpp"

namespace BOOM {

  class DynamicRegressionArPosteriorSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model to be posterior sampled.
    //   siginv_priors: The prior distribution on the (reciprocal) residual
    //     variance of each coefficient.
    //   seeding_rng: The random number generator used to seed the RNG for this
    //     sampler.
    DynamicRegressionArPosteriorSampler(
        DynamicRegressionArStateModel *model,
        const std::vector<Ptr<GammaModelBase>> &siginv_priors,
        RNG &seeding_rng = GlobalRng::rng);

    DynamicRegressionArPosteriorSampler * clone_to_new_host(
        Model *new_host) const override;

    double logpri() const override;
    void draw() override;

   private:
    DynamicRegressionArStateModel *model_;
    std::vector<Ptr<ArPosteriorSampler>> samplers_;
  };

}  //  namespace BOOM
#endif  //  BOOM_DYNAMIC_REGRESSION_AUTOREGRESSIVE_POSTERIOR_SAMPLER_HPP_

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

#ifndef BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_GAUSSIAN_REGRESSION_SAMPLER_HPP_
#define BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_GAUSSIAN_REGRESSION_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Hierarchical/HierarchicalGaussianRegressionModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class HierarchicalGaussianRegressionSampler : public PosteriorSampler {
   public:
    HierarchicalGaussianRegressionSampler(
        HierarchicalGaussianRegressionModel *model,
        const Ptr<GammaModelBase> &residual_precision_prior,
        RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

   private:
    HierarchicalGaussianRegressionModel *model_;
    Ptr<GammaModelBase> residual_variance_prior_;
    GenericGaussianVarianceSampler residual_variance_sampler_;
  };

}  // namespace BOOM

#endif  // BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_GAUSSIAN_REGRESSION_SAMPLER_HPP_

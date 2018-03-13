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

#ifndef BOOM_HIERARCHICAL_POISSON_POSTERIOR_SAMPLER_HPP_
#define BOOM_HIERARCHICAL_POISSON_POSTERIOR_SAMPLER_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/Hierarchical/HierarchicalPoissonModel.hpp"

namespace BOOM {

  class HierarchicalPoissonSampler : public PosteriorSampler {
   public:
    // The top level of the HierarchicalPoissonModel is a gamma
    // distribution, which is assumed to model the Poisson rate
    // parameters in each group of the heirarchy.  The traditional
    // parameterization of the gamma distribution is gamma(a, b),
    // where the mean is a/b and the variance is a/b^2.
    //
    // This sampler expects a prior on the mean a/b and an independent
    // prior on a.  If we write mu = a/b then the variance of the
    // gamma distribution is is mu^2 / a, so 'a' acts kind of like a
    // prior "sample size" that determines the amount of shrinkage of
    // the group level means around the prior mean mu.
    //
    // Args:
    //   model: The model whose parameters are to be to be sampled
    //     from their posterior distribution.
    //   gamma_mean_prior: Prior distribution on the mean of the gamma
    //     distribution: a/b.
    //   gamma_sample_size_prior: Prior distribution on the shape
    //     parameter of the gamma distribution: a.
    HierarchicalPoissonSampler(HierarchicalPoissonModel *model,
                               const Ptr<DoubleModel> &gamma_mean_prior,
                               const Ptr<DoubleModel> &gamma_sample_size_prior,
                               RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;

   private:
    HierarchicalPoissonModel *model_;
    Ptr<DoubleModel> gamma_mean_prior_;
    Ptr<DoubleModel> gamma_sample_size_prior_;
  };

}  // namespace BOOM

#endif  //  BOOM_HIERARCHICAL_POISSON_POSTERIOR_SAMPLER_HPP_

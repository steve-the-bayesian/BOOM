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

#ifndef BOOM_GAUSSIAN_LINEAR_BART_POSTERIOR_SAMPLER_HPP
#define BOOM_GAUSSIAN_LINEAR_BART_POSTERIOR_SAMPLER_HPP

#include "Models/Bart/GaussianLinearBartModel.hpp"
#include "Models/Bart/PosteriorSamplers/GaussianBartPosteriorSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // Combines the regression component of the GaussianLinearBartModel
  // with a spike and slab prior, and the Bart part of the model with a
  // traditional Bart prior.
  //
  // At each stage of the MCMC, the Bart predictions are subtracted from
  // the data for the regression model, and vice versa.
  class GaussianLinearBartPosteriorSampler : public PosteriorSampler {
   public:
    GaussianLinearBartPosteriorSampler(
        GaussianLinearBartModel *model,
        const ZellnerPriorParameters &regression_prior,
        const BartPriorParameters &bart_prior,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void sample_regression_posterior();
    void sample_bart_posterior();
    void adjust_regression_residuals();
    void adjust_bart_residuals();

   private:
    GaussianLinearBartModel *model_;
    bool first_time_for_regression_;
    Ptr<GaussianBartPosteriorSampler> bart_sampler_;
    bool first_time_for_bart_;
  };

}  // namespace BOOM

#endif  //  BOOM_GAUSSIAN_LINEAR_BART_POSTERIOR_SAMPLER_HPP

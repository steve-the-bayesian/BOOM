// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MVREG_SAMPLER_HPP
#define BOOM_MVREG_SAMPLER_HPP
#include "Models/Glm/MultivariateRegression.hpp"

namespace BOOM {

  class MultivariateRegressionSampler : public PosteriorSampler {
   public:
    // Assumes the prior distribution Beta|Sigma ~ N(B, Sigma \otimes I/kappa)
    // and Sigma^{-1} ~ Wishart(prior_df/2, SS/2);
    //
    // Args:
    //   model:  The model that needs posterior sampling.
    //   coefficient_prior_mean: Rows correspond to predictor variables.
    //     Columns to response variables.
    //   coefficient_prior_sample_size: Prior sample size for estimating the
    //     coefficients.  'kappa' in the formula above.
    //   prior_df:  Prior sample size for estimating the residual variance.
    //   residual_variance_guess:  Prior guess at the residual variance.
    //   rng: Random number generator used to seed this rng() method for this
    //     object.
    MultivariateRegressionSampler(
        MultivariateRegressionModel *model,
        const Matrix &coefficient_prior_mean,
        double coefficient_prior_sample_size,
        double prior_df,
        const SpdMatrix &residual_variance_guess,
        RNG &seeding_rng = GlobalRng::rng);

    double logpri() const override;
    void draw() override;
    void draw_Beta();
    void draw_Sigma();

   private:
    MultivariateRegressionModel *model_;
    SpdMatrix SS_;
    double prior_df_;
    SpdMatrix Ominv_;
    double ldoi_;
    Matrix beta_prior_mean_;
  };

}  // namespace BOOM
#endif  // BOOM_MVREG_SAMPLER_HPP

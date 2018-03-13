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

#ifndef BOOM_REGRESSION_CONJUGATE_SAMPLER_HPP
#define BOOM_REGRESSION_CONJUGATE_SAMPLER_HPP

#include "Models/GammaModel.hpp"
#include "Models/Glm/MvnGivenXandSigma.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class RegressionConjSampler : public PosteriorSampler {
    // for drawing p(beta, sigma^2 | y)
    // prior is p(beta | sigma^2, X) = N(b0, sigsq * XTX/kappa)
    //          p(sigsq | X) = Gamma(prior_df/2, prior_ss/2)
   public:
    RegressionConjSampler(RegressionModel *model,
                          const Ptr<MvnGivenXandSigma> &coefficient_prior,
                          const Ptr<GammaModelBase> &residual_precision_prior,
                          RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }

    const Vector &b0() const;
    double kappa() const;
    double prior_df() const;
    double prior_ss() const;

   private:
    RegressionModel *m_;
    Ptr<MvnGivenXandSigma> mu_;
    Ptr<GammaModelBase> siginv_;
    Vector beta_tilde;
    SpdMatrix ivar;
    double SS, DF;
    GenericGaussianVarianceSampler sigsq_sampler_;
    void set_posterior_suf();
  };
}  // namespace BOOM
#endif  // BOOM_REGRESSION_CONJUGATE_SAMPLER_HPP

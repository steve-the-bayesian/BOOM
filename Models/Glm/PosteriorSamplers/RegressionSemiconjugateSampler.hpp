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

#ifndef BOOM_REGRESSION_SEMICONJUGATE_POSTERIOR_SAMPLER_HPP_
#define BOOM_REGRESSION_SEMICONJUGATE_POSTERIOR_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionCoefficientSampler.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // A regression sampler with a semi-conjugate prior p(beta, sigsq) = p(beta) *
  // p(sigsq)
  class RegressionSemiconjugateSampler : public PosteriorSampler {
   public:
    RegressionSemiconjugateSampler(
        RegressionModel *model, const Ptr<MvnBase> &coefficient_prior,
        const Ptr<GammaModelBase> &residual_precision_prior,
        RNG &seeding_rng = GlobalRng::rng);

    // Truncate the support of the residual standard deviation ("sigma") to (0,
    // sigma_max).
    void set_sigma_max(double sigma_max);

    void draw() override;
    double logpri() const override;

    void draw_beta_given_sigma();
    void draw_sigma_given_beta();

    bool can_find_posterior_mode() const override { return true; }
    bool can_evaluate_log_prior_density() const override { return true; }
    bool can_increment_log_prior_gradient() const override { return true; }

    void find_posterior_mode(double epsilon = 1e-5) override;
    double log_prior_density(const ConstVectorView &parameters) const override;
    double increment_log_prior_gradient(const ConstVectorView &parameters,
                                        VectorView gradient) const override;
    double log_prior(const Vector &parameters, Vector &gradient,
                     Matrix &hessian, uint nd) const;

   private:
    RegressionModel *model_;
    Ptr<MvnBase> beta_prior_;
    Ptr<GammaModelBase> siginv_prior_;

    RegressionCoefficientSampler beta_sampler_;
    GenericGaussianVarianceSampler sigsq_sampler_;
  };
}  // namespace BOOM

#endif  // BOOM_REGRESSION_SEMICONJUGATE_POSTERIOR_SAMPLER_HPP_

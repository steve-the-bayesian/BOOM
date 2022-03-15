// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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
#ifndef BOOM_ZERO_MEAN_GAUSSIAN_CONJ_SAMPLER_HPP_
#define BOOM_ZERO_MEAN_GAUSSIAN_CONJ_SAMPLER_HPP_

#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class ZeroMeanGaussianModel;

  class ZeroMeanGaussianConjSampler : public PosteriorSampler {
   public:
    ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel *model,
                                const Ptr<GammaModelBase> &siginv_prior,
                                RNG &seeding_rng = GlobalRng::rng);
    ZeroMeanGaussianConjSampler(ZeroMeanGaussianModel *model,
                                double df,
                                double sigma_guess,
                                RNG &seeding_rng = GlobalRng::rng);

    ZeroMeanGaussianConjSampler *clone() const;
    ZeroMeanGaussianConjSampler *clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;

    double sigma_prior_guess() const;
    double sigma_prior_sample_size() const;
    void set_sigma_upper_limit(double sigma_upper_limit);

    // The posterior mode is with respect to sigma^2, not 1 / sigma^2.
    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }
    bool can_evaluate_log_prior_density() const override { return true; }
    bool can_increment_log_prior_gradient() const override { return true; }

    double log_prior_density(const ConstVectorView &parameters) const override;
    double increment_log_prior_gradient(const ConstVectorView &parameters,
                                        VectorView gradient) const override;

    // Rturns the log prior density, and the first or second
    // derivatives, if requested.
    // Args:
    //   sigsq: The value of the variance parameter at which to
    //     evaluate the log density.
    //   d1: Ignored if nullptr.  Otherwise, input value is
    //     incremented by the first derivative of log density with
    //     respect to sigma^2.
    //   d2: Ignored if nullptr.  Otherwise, input value is
    //     incremented by the second derivative of log prior density
    //     with respect to sigma^2.
    // Returns:
    //   The un-normalized log posterior density (log likelihood +
    //   log prior) evaluated at sigsq.
    double log_prior(double sigsq, double *d1, double *d2) const;

    // Returns the log posterior with respect to sigma^2, including
    // derivatives with respect to sigma^2.
    // Args:
    //   sigsq: The value of the variance parameter at which to
    //     evaluate the log density.
    //   d1: Input value is not used.  If nd > 0 then on output d1
    //     contains the first derivative of log density with respect
    //     to sigma^2.
    //   d2: Input value is not used.  If nd > 1 then on output d2
    //     contains the second derivative of log density with respect
    //     to sigma^2.
    // Returns:
    //   The un-normalized log posterior density (log likelihood +
    //   log prior) evaluated at sigsq.
    double log_posterior(double sigsq, double &d1, double &d2, uint nd) const;

   private:
    ZeroMeanGaussianModel *model_;
    Ptr<GammaModelBase> precision_prior_;
    GenericGaussianVarianceSampler variance_sampler_;
  };

}  // namespace BOOM
#endif  // BOOM_ZERO_MEAN_GAUSSIAN_CONJ_SAMPLER_HPP_

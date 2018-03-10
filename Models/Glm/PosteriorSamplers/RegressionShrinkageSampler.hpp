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

#ifndef BOOM_REGRESSION_SHRINKAGE_SAMPLER_HPP_
#define BOOM_REGRESSION_SHRINKAGE_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/GaussianModelBase.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // Posterior sampling for regression models under a prior that shrinks groups
  // of coefficients together.  The model is
  //
  //                y[i] ~ N(beta.dot(x), sigma^2)
  //         1 / sigma^2 ~ Gamma(df/2, ss/2)
  //      group(beta, 0) ~ N(b0, v0)
  //      group(beta, 1) ~ N(b1, v1)
  //      ...
  //
  // In this notation, group(beta, k) ~ N(m, v) means that the subset of beta
  // belonging to group k is IID according to the specified normal distribution.
  // Further priors may be placed on the parameters of the normal distributions
  // governing shrinkage within a group.
  class RegressionShrinkageSampler : public PosteriorSampler {
   public:
    // A class describing the set of regression coefficients that are
    // exchangeable.
    class CoefficientGroup {
     public:
      // Args:
      //   prior: The prior distribution describing the coefficients in this
      //     group.  If the prior is to be updated as part of an MCMC algorithm,
      //     its posterior sampler should be set prior to passing it to this
      //     constructor.
      //   indices: A set of integers indicating which components of the vector
      //     of regression coefficients belong to this coefficient group.
      CoefficientGroup(const Ptr<GaussianModelBase> &prior,
                       const std::vector<int> &indices);

      double prior_mean() const { return prior_->mu(); }
      double prior_variance() const { return prior_->sigsq(); }
      double prior_sd() const { return prior_->sigma(); }

      const std::vector<int> &indices() const { return indices_; }

      // Clear the sufficient statistics from the prior model, and leave the
      // prior model with a set of sufficient statistics computed from the
      // relevant elements from the argument.
      //
      // Args:
      //   coefficients: The full vector of coefficients.
      void refresh_sufficient_statistics(const Vector &coefficients);

      void sample_posterior() { prior_->sample_posterior(); }

      double log_prior(double b) const { return prior_->logp(b); }

      double log_hyperprior() const { return prior_->logpri(); }

     private:
      Ptr<GaussianModelBase> prior_;
      std::vector<int> indices_;
    };  // End of the CoefficientGroup class.
    //----------------------------------------------------------------------
    // Args:
    //   model:  The model whose coefficients are to be sampled using MCMC.
    //   coefficient_groups: Defines which coefficients are to be viewed as
    //     exchangeable.
    RegressionShrinkageSampler(
        RegressionModel *model,
        const Ptr<GammaModelBase> &residual_precision_prior,
        const std::vector<CoefficientGroup> &groups,
        RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    void draw_coefficients();
    void draw_hyperparameters();
    void draw_residual_variance();

    double logpri() const override;

    Vector prior_mean() const;
    Vector prior_precision_diagonal() const;

   private:
    RegressionModel *model_;
    GenericGaussianVarianceSampler variance_sampler_;
    std::vector<CoefficientGroup> groups_;
  };

}  // namespace BOOM
#endif  //  BOOM_REGRESSION_SHRINKAGE_SAMPLER_HPP_

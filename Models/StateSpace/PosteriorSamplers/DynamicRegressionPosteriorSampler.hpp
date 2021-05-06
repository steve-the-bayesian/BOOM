// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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
#ifndef BOOM_DYNAMIC_REGRESSION_POSTERIOR_SAMPLER_HPP_
#define BOOM_DYNAMIC_REGRESSION_POSTERIOR_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"

namespace BOOM {

  // A posterior sampler for a dynamic regression state model.  In the model,
  // each coefficient moves according to an independent random walk with
  // Gaussian noise.  This class models the variance for each coefficient using
  // an independent conjugate prior.
  class DynamicRegressionIndependentPosteriorSampler : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model to be managed by this sampler.
    //   innovation_precision_priors: The prior distribution for the precision
    //     (reciprocal variance) describing the innovations for each
    //     coefficient.  If the vector contains a single object it will be
    //     copied the appropriate number of times.
    //   seeding_rng: The random number generator used to seed this sampler's
    //     RNG.
    DynamicRegressionIndependentPosteriorSampler(
        DynamicRegressionStateModel *model,
        const std::vector<Ptr<GammaModelBase>> &innovation_precision_priors,
        RNG &seeding_rng = GlobalRng::rng);

    DynamicRegressionIndependentPosteriorSampler *clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;

    // Sets the maximum value for the standard deviation describing a
    // coefficient's innovations.
    // Args:
    //   coefficient: The coefficient whose standard devitation is to be
    //     bounded.
    //   value:  The bound to place on the standard devitation.
    void set_sigma_max(int coefficient, double value);

   private:
    DynamicRegressionStateModel *model_;
    std::vector<Ptr<GammaModelBase>> priors_;
    std::vector<GenericGaussianVarianceSampler> samplers_;
  };

  // The prior in this sampler is a hierarchical model that pools information
  // about the size of the precision for each coefficient's innovations.  The
  // model is beta[i, t + 1] = beta[i, t] + error[i, t], with error[i, t] IID
  // N(0, sigma[i]^2) and 1/sigma[i]^2 ~ Gamma(a, b).  A prior distribution is
  // placed on 'a' and 'b' and these values are learned from the data.
  //
  // This model is appropriate for settings where there are many coefficients.
  // If there are only 1 or a handful of coefficients then there is not much to
  // be gained from the hierarchy.
  class DynamicRegressionPosteriorSampler : public PosteriorSampler {
   public:
    // Args:
    //   model: The state model to be managed.
    //   siginv_prior: A prior distribution for the precision (reciprocal
    //     variance) of the innovation for each coefficient.  This class will
    //     handle setting data for siginv_prior, but its posterior sampler
    //     should be set externally by the user.
    //   seeding_rng: The random number generator used to create the seed for
    //     the RNG embedded in this object.
    DynamicRegressionPosteriorSampler(DynamicRegressionStateModel *model,
                                      const Ptr<GammaModel> &siginv_prior,
                                      RNG &seeding_rng = GlobalRng::rng);

    DynamicRegressionPosteriorSampler *clone_to_new_host(
        Model *new_host) const override;

    // By default the class will take control of siginv_prior, updating it when
    // draw() is called, and adding its contribution to logpri().  If you want
    // to avoid this behavior and manage siginv_prior outside of this class then
    // call handle_siginv_prior_separately().
    void handle_siginv_prior_separately();

    // logpri() returns the prior with respect to sigma[i].  It does not return
    // the hyperprior of siginv_prior.  A separate call to
    // siginv_prior->logpri() will return that.
    double logpri() const override;

    // Draw and set new values for the variance parameters of each coefficient
    // in the dynamic regression model.  Also call the sample_posterior() method
    // on siginv_prior_, unless handle_siginv_prior_separately() has been set.
    void draw() override;

    void set_sigma_max(double sigma_max) {
      sigsq_sampler_.set_sigma_max(sigma_max);
    }

   private:
    DynamicRegressionStateModel *model_;
    Ptr<GammaModel> siginv_prior_;
    GenericGaussianVarianceSampler sigsq_sampler_;
    bool handle_siginv_prior_separately_;
  };

}  // namespace BOOM

#endif  //  BOOM_DYNAMIC_REGRESSION_POSTERIOR_SAMPLER_HPP_

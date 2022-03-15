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

#ifndef BOOM_ARP_POSTERIOR_SAMPLER_HPP_
#define BOOM_ARP_POSTERIOR_SAMPLER_HPP_

#include "Models/GammaModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/HierarchicalPosteriorSampler.hpp"
#include "Models/TimeSeries/ArModel.hpp"

namespace BOOM {

  // A sampler for an AR(p) process, assuming a uniform prior over the
  // AR coefficients with support over the stationary region, and an
  // inverse Gamma prior on innovation variance.
  class ArPosteriorSampler : public HierarchicalPosteriorSampler {
   public:
    ArPosteriorSampler(ArModel *model, const Ptr<GammaModelBase> &siginv_prior,
                       RNG &seeding_rng = GlobalRng::rng);

    // The 'draw' method will make several attempts to simulate AR
    // coefficients directly from the posterior distribution of a
    // regression model conditional on sigma.  If the maximum number
    // of proposals is exceeded then a series of univariate draws will
    // be made starting from the current value of the AR coefficients.
    void draw() override;
    void draw_model_parameters(Model &model) override;
    void draw_model_parameters(ArModel &model);

    // Draw the residual variance for the specified model.
    // Args:
    //   model:  The ArModel that needs its residual variance parameter drawn.
    //   sigma_guess_scale_factor: A factor by which to multiply the estimated
    //     standard deviation implied by siginv_prior.  Most of the time this
    //     will just be 1.0, but in cases where the ArPosteriorSampler is
    //     applied to multiple models that differ only by a scale factor, this
    //     argument allows multiple models to be used with a single sampler.
    void draw_sigma(ArModel &model, double sigma_guess_scale_factor = 1.0);
    void draw_phi(ArModel &model);

    // Uses a univariate slice sampler to draw each component of phi
    // given the others.
    void draw_phi_univariate(ArModel &model);

    // Returns -infinity if the coefficients are outside of the legal
    // range.  Returns logp(siginv) otherwise.
    double logpri() const override;
    double log_prior_density(const Model &model) const override;
    double log_prior_density(const ArModel &model) const;

    void set_max_number_of_regression_proposals(int number_of_proposals);

    void set_sigma_upper_limit(double max_sigma);

    const Ptr<GammaModelBase> &residual_precision_prior() const {
      return siginv_prior_;
    }

   private:
    ArModel *model_;
    Ptr<GammaModelBase> siginv_prior_;
    int max_number_of_regression_proposals_;
    GenericGaussianVarianceSampler sigsq_sampler_;
  };

}  // namespace BOOM

#endif  //  BOOM_ARP_POSTERIOR_SAMPLER_HPP_

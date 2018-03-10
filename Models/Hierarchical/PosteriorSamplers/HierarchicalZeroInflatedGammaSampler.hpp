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

#ifndef BOOM_HIERARCHICAL_ZERO_INFLATED_GAMMA_SAMPLER_HPP_
#define BOOM_HIERARCHICAL_ZERO_INFLATED_GAMMA_SAMPLER_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/Hierarchical/HierarchicalZeroInflatedGammaModel.hpp"
#include "Models/PosteriorSamplers/BetaPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/GammaPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/ZeroInflatedGammaPosteriorSampler.hpp"

namespace BOOM {

  class HierarchicalZeroInflatedGammaSampler : public PosteriorSampler {
   public:
    // The constructor takes the model to be sampled (as usual) and six
    // independent distributions defining the prior.
    // Args:
    //   model:  The model managed by this sampler.
    //   gamma_mean_mean_prior: The prior distribution for the mu_mean
    //     parameter.  This parameter is the large sample grand mean
    //     of the model, across all groups.  If this parameter is
    //     large, then group means will tend to be large.
    //   gamma_mean_shape_prior: The prior distribution for the
    //     mu_shape parameter.  This parameter governs the spread of
    //     the group-level means.  If this parameter is large then the
    //     group-level means (mu[i]) tend to be homogenous across
    //     groups.
    //   gamma_shape_mean_prior: The prior distribution for the a_mean
    //     parameter, which is governs the precision of individual
    //     observations around their group means.  If this parameter
    //     is large then group-level shape parameters (a[i]) will tend
    //     to be large, implying that nonzero observations in the data
    //     are relatively tight around the group means.
    //   gamma_shape_shape_prior: The prior distribution for the
    //     a_shape parameter, which governs the group-to-group
    //     variation in the shape parameters.  If this parameter is
    //     large then group-level shape parameters (a[i]) will tend to
    //     be homogeneous across groups.
    //   positive_probability_mean_prior: The prior distribution for
    //     the group level means of the positive probability
    //     parameter.
    //   positive_probability_sample_size_prior: Prior distribution
    //     for positive_probability_sample_size parameter.  If this
    //     parameter is large then the positive probabilities will be
    //     homogeneous across groups.
    HierarchicalZeroInflatedGammaSampler(
        HierarchicalZeroInflatedGammaModel *model,
        const Ptr<DoubleModel> &gamma_mean_mean_prior,
        const Ptr<DoubleModel> &gamma_mean_shape_prior,
        const Ptr<DoubleModel> &gamma_shape_mean_prior,
        const Ptr<DoubleModel> &gamma_shape_shape_prior,
        const Ptr<DoubleModel> &positive_probability_mean_prior,
        const Ptr<DoubleModel> &positive_probability_sample_size_prior,
        RNG &seeding_rng = GlobalRng::rng);
    double logpri() const override;
    void draw() override;

   private:
    // Check that a posterior sampler has been assigned to
    // *data_model.  If not, assign one.
    void ensure_posterior_sampling_method(ZeroInflatedGammaModel *data_model);

    HierarchicalZeroInflatedGammaModel *model_;
    Ptr<DoubleModel> gamma_mean_mean_prior_;
    Ptr<DoubleModel> gamma_mean_shape_prior_;
    Ptr<DoubleModel> gamma_shape_mean_prior_;
    Ptr<DoubleModel> gamma_shape_shape_prior_;
    Ptr<DoubleModel> positive_probability_mean_prior_;
    Ptr<DoubleModel> positive_probability_sample_size_prior_;

    // Responsible for drawing mu_mean, and mu_shape.
    Ptr<GammaPosteriorSampler> gamma_mean_sampler_;

    // Responsible for drawing a_mean, and a_shape.
    Ptr<GammaPosteriorSampler> gamma_shape_sampler_;

    // Responsible for drawing positive_probability_mean and
    // positive_probability_sample_size.
    Ptr<BetaPosteriorSampler> positive_probability_prior_sampler_;
  };

}  // namespace BOOM

#endif  // BOOM_HIERARCHICAL_ZERO_INFLATED_POISSON_SAMPLER_HPP_

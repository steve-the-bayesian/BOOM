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

#ifndef BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_POSTERIOR_SAMPLER_HPP_
#define BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_POSTERIOR_SAMPLER_HPP_

#include "Models/ModelTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class HierarchicalPosteriorSampler : public PosteriorSampler {
   public:
    explicit HierarchicalPosteriorSampler(RNG &seeding_rng = GlobalRng::rng);

    // Child classes should be capable of drawing model parameters in models
    // with no data.
    virtual void draw_model_parameters(Model &model) = 0;

    virtual double log_prior_density(const Model &model) const = 0;
    using PosteriorSampler::log_prior_density;
  };

  class ConjugateHierarchicalPosteriorSampler
      : public HierarchicalPosteriorSampler {
   public:
    explicit ConjugateHierarchicalPosteriorSampler(
        RNG &seeding_rng = GlobalRng::rng);

    // Evaluates the log of the marginal density function
    //   p(y) = \int p(y | theta) p(theta).
    //
    // Here y is the data contained in dp, theta are the paramters in model, and
    // p(theta) is prior distribution managed by this object.
    //
    // Args:
    //   dp:  Pointer to the data (y) where the density is to be evaluated.
    //   model: A model object from a model family conjugate to this posterior
    //     sampler.
    //
    // Returns: the log marginal density at y.
    virtual double log_marginal_density(const Ptr<Data> &dp,
                                        const ConjugateModel *model) const = 0;
  };

}  // namespace BOOM

#endif  // BOOM_POSTERIOR_SAMPLERS_HIERARCHICAL_POSTERIOR_SAMPLER_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_INDEPENDENT_MVN_VAR_SAMPLER_HPP_
#define BOOM_INDEPENDENT_MVN_VAR_SAMPLER_HPP_

#include <vector>
#include "Models/GammaModel.hpp"
#include "Models/IndependentMvnModel.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  // A posterior sampler for IndependentMvnModel objects where the
  // 'mean' parameter is assumed known.
  class IndependentMvnVarSampler : public PosteriorSampler {
   public:
    // Args:
    //   model: The model whose diagonal variance parameters are to be
    //     posterior sampled.  The mean parameter of this model should
    //     be set to the known value.
    //   priors: The independent prior distributions for the diagonal
    //     elements of the (diagonal) precision matrix.
    //   sd_max_values: The largest admissible values for the standard
    //     deviations.  This can be an empty Vector, as a signal that
    //     the upper support is unbounded.  Otherwise the length of
    //     sd_max_values must match the model->dim().  Negative values
    //     are not permitted, but zero values are (indicating that
    //     specific elements should be set to zero), as are infinite
    //     values (indicating that specific elements have unbounded
    //     support).
    IndependentMvnVarSampler(IndependentMvnModel *model,
                             const std::vector<Ptr<GammaModelBase>> &priors,
                             Vector sd_max_values = Vector(),
                             RNG &seeding_rng = GlobalRng::rng);
    IndependentMvnVarSampler *clone_to_new_host(Model *new_host) const override;

    double logpri() const override;
    void draw() override;

    // Truncate the support for the standard deviations to [0, sigma_max[i]).
    void set_sigma_max(const Vector &sigma_max);

   private:
    IndependentMvnModel *model_;
    std::vector<Ptr<GammaModelBase>> priors_;
    std::vector<GenericGaussianVarianceSampler> samplers_;
  };

}  // namespace BOOM
#endif  // BOOM_INDEPENDENT_MVN_VAR_SAMPLER_HPP_

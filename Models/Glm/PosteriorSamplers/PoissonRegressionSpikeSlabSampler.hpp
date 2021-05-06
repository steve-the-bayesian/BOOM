// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#ifndef BOOM_GLM_POISSON_REGRESSION_SPIKE_SLAB_POSTERIOR_SAMPLER_HPP_
#define BOOM_GLM_POISSON_REGRESSION_SPIKE_SLAB_POSTERIOR_SAMPLER_HPP_

#include "Models/Glm/PoissonRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionAuxMixSampler.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"

namespace BOOM {

  // A spike-and-slab sampler for Poisson regression models.
  class PoissonRegressionSpikeSlabSampler
      : public PoissonRegressionAuxMixSampler {
   public:
    // Args:
    //   model:  The model to be posterior sampled.
    //   slab_prior: The prior distribution for the Poisson regression
    //     coefficients, conditional on inclusion.
    //   spike_prior: The prior on which coefficients should be
    //     included.
    //   number_of_threads: The number of threads to use for data
    //     augmentation.
    PoissonRegressionSpikeSlabSampler(
        PoissonRegressionModel *model, const Ptr<MvnBase> &slab_prior,
        const Ptr<VariableSelectionPrior> &spike_prior,
        int number_of_threads = 1, RNG &seeding_rng = GlobalRng::rng);

    PoissonRegressionSpikeSlabSampler *clone_to_new_host(
        Model *new_host) const override;

    void draw() override;
    double logpri() const override;

    // If tf == true then draw_model_indicators is a no-op.  Otherwise
    // model indicators will be sampled each iteration.
    void allow_model_selection(bool tf);

    // In very large problems you may not want to sample every element
    // of the inclusion vector each time.  If max_flips is set to a
    // positive number then at most that many randomly chosen
    // inclusion indicators will be sampled.
    void limit_model_selection(int max_flips);

    // Sets the coefficients in model_ to their posterior mode, and
    // saves the value of the un-normalized log-posterior at the
    // mode.  The optimization is with respect to coefficients that
    // are "in" the model.  Dropped coefficients will remain zero.
    void find_posterior_mode(double epsilon = 1e-5) override;

    bool can_find_posterior_mode() const override { return true; }

    double log_posterior_at_mode() const { return log_posterior_at_mode_; }

   private:
    PoissonRegressionModel *model_;
    SpikeSlabSampler sam_;
    Ptr<MvnBase> slab_prior_;
    Ptr<VariableSelectionPrior> spike_prior_;
    double log_posterior_at_mode_;
  };

}  // namespace BOOM

#endif  //  BOOM_GLM_POISSON_REGRESSION_SPIKE_SLAB_POSTERIOR_SAMPLER_HPP_

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

#ifndef BOOM_GLM_SPIKE_SLAB_SAMPLER_HPP_
#define BOOM_GLM_SPIKE_SLAB_SAMPLER_HPP_

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/MvnBase.hpp"

namespace BOOM {

  // A class to manage the elements of spike-and-slab posterior
  // sampling common to GlmModel objects.  This class does not inherit
  // from PosteriorSampler because it is intended to be an element of
  // a model-specific PosteriorSampler class that can impute the
  // latent variables needed to turn the GLM into a Gaussian problem.
  //
  // If the GlmModel has a residual variance parameter, this draw
  // conditions on it.  The alternative is to integrate it out, which
  // changes the form of log_model_prob from something that looks like
  // a normal to something that looks like a T.
  class SpikeSlabSampler {
   public:
    SpikeSlabSampler(GlmModel *model, const Ptr<MvnBase> &slab_prior,
                     const Ptr<VariableSelectionPrior> &spike_prior);
    double logpri() const;

    // Performs one MCMC sweep along the inclusion indicators for the
    // managed GlmModel.
    // Args:
    //   rng:  The uniform random number generator.
    //   suf: Complete data sufficient statistics corresponding to
    //     X'WX and X'Wy for appropriate values of W.
    //   sigsq: Models that have a residual variance parameter (so the
    //     variance of each normal deviate is w[i] * sigsq) can
    //     provide it here.  Models that do not have a sigsq parameter
    //     should pass sigsq = 1.0.
    void draw_model_indicators(RNG &rng, const WeightedRegSuf &suf,
                               double sigsq = 1.0);

    // Draws the set of included Glm coefficients given complete data
    // sufficient statistics.
    void draw_beta(RNG &rng, const WeightedRegSuf &suf, double sigsq = 1.0);

    // If tf == true then draw_model_indicators is a no-op.  Otherwise
    // model indicators will be sampled each iteration.
    void allow_model_selection(bool tf);

    // In very large problems you may not want to sample every element
    // of the inclusion vector each time.  If max_flips is set to a
    // positive number then at most that many randomly chosen
    // inclusion indicators will be sampled.
    void limit_model_selection(int max_flips);

   private:
    // Compute the log of the marginal posterior probability of model 'g'.
    // Args:
    //   g: The set of included coefficients defining the model.
    //   suf:  The set of complete data sufficient statistics.
    //   sigsq: If the model has a residual variance parameter that is
    //     not reflected in 'suf' provide it here.  Models that do not
    //     have a separate residual variance parameter should use
    //     sigsq = 1.0.
    double log_model_prob(const Selector &g, const WeightedRegSuf &suf,
                          double sigsq) const;

    // A single MCMC step for a single position in the set of
    // coefficient indicators 'g'.
    // Args:
    //   rng:  A Uniform(0,1) random number generator.
    //   g: The set of included coefficients defining the model.  One
    //     element of 'g' may be changed.
    //   which_variable:  The position in 'g' to consider changing.
    //   logp_old: The value of log_model_prob(g) prior to calling
    //     this function.
    //   suf:  The set of complete data sufficient statistics.
    //   sigsq: If the model has a residual variance parameter that is
    //     not reflected in 'suf' provide it here.  Models that do not
    //     have a separate residual variance parameter should use
    //     sigsq = 1.0.
    double mcmc_one_flip(RNG &rng, Selector &g, int which_variable,
                         double logp_old, const WeightedRegSuf &suf,
                         double sigsq);

    GlmModel *model_;
    Ptr<MvnBase> slab_prior_;
    Ptr<VariableSelectionPrior> spike_prior_;
    int max_flips_;
    bool allow_model_selection_;
  };

}  // namespace BOOM

#endif  // BOOM_GLM_SPIKE_SLAB_SAMPLER_HPP_

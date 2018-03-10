// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#ifndef BOOM_NONCONJUGATE_REGRESSION_SAMPLER_HPP_
#define BOOM_NONCONJUGATE_REGRESSION_SAMPLER_HPP_

#include <functional>

#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

#include "Models/GammaModel.hpp"
#include "Models/VectorModel.hpp"

#include "Models/Glm/RegressionModel.hpp"

#include "Samplers/MH_Proposals.hpp"
#include "Samplers/MetropolisHastings.hpp"
#include "Samplers/MoveAccounting.hpp"
#include "Samplers/UnivariateSliceSampler.hpp"

namespace BOOM {

  // A posterior sampler for a regression model with a prior
  // constraint that all coefficients are positive.  Posterior
  // sampling mixes at random between a MH algorithm based on
  // truncating a nearly normal posterior, and a univariate slice
  // sampler.
  class NonconjugateRegressionSampler : public PosteriorSampler {
   public:
    // Args:
    //   model: The model for which to create a posterior sampler.
    //   beta_prior: The prior distribution for the regression
    //     coefficients.
    //   residual_precision_prior: The prior distribution for the
    //     inverse of the residual variance.
    //   seeding_rng: The random number generator used to generate the
    //     seed for this sampler's RNG.
    NonconjugateRegressionSampler(
        RegressionModel *model, const Ptr<LocationScaleVectorModel> &beta_prior,
        const Ptr<GammaModelBase> &residual_precision_prior,
        RNG &seeding_rng = GlobalRng::rng);

    // If the prior distribution limits the support of the regression
    // coefficients, then a rectangle of support can be specified here
    // so that the slice sampler won't spend a bunch of time looking
    // around in regions of zero prior probability.  Only rectangular
    // regions are supported.
    void set_slice_sampler_limits(const Vector &lower, const Vector &upper);

    void draw() override;
    double logpri() const override;

    void draw_coefficients();
    void draw_sigsq();

    // Methods used to implement draw_coefficients.
    void draw_using_mh();
    void draw_using_slice();
    void refresh_mh_proposal_distribution();

    // Returns a callback the computes log p(beta | y, sigsq).
    std::function<double(const Vector &beta)> beta_log_posterior_callback();

    // Keeps track of how many times each sampler was called, how many
    // rejections MH had, and how much time each sampler consumed.
    const MoveAccounting &move_accounting() const { return move_accounting_; }

   private:
    RegressionModel *model_;
    Ptr<LocationScaleVectorModel> beta_prior_;
    Ptr<GammaModelBase> residual_precision_prior_;

    GenericGaussianVarianceSampler residual_variance_sampler_;

    Ptr<MvtIndepProposal> mh_proposal_;
    MetropolisHastings mh_sampler_;
    UnivariateSliceSampler slice_sampler_;

    // To keep track of how many moves there have been of each type,
    // how long they have taken, and how much movement took place.
    MoveAccounting move_accounting_;

    enum SamplingMethod { METROPOLIS, SLICE };
    // For the first 100 iterations samplers are selected at random.
    // Afterwards the MH algorithm is selected in proportion to its
    // probability of accepting.
    SamplingMethod select_sampling_method();
  };

}  // namespace BOOM

#endif  // BOOM_NONCONJUGATE_REGRESSION_SAMPLER_HPP_

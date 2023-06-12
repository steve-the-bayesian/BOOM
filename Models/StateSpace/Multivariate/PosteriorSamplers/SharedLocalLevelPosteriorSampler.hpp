#ifndef BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/MvnBase.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"

namespace BOOM {

  // A posterior sampler for the shared local level model.  The local level
  // model assumes k "factors" in the vector alpha are linearly related to m
  // observed series in y:
  //
  //          y[t] = Z * alpha[t] + epsilon[t]
  //    alpha[t+1] =     alpha[t] + eta[t]
  //
  // The parameters in this model are Z, which contains a constrained set of
  // regression coefficients, and the variance matrix for eta.
  //
  // The prior assumed here is a product of independent inverse gamma
  // distributions on the innovation variances, with a sequence of independent
  // regression priors on Z.
  //
  // This model contains both scale and rotation indeterminacies, because Z *
  // alpha = Z * M * M.inv() * alpha for any matrix M.
  //
  // The scale issue can be resolved by fixing the innovation variance, e.g. at
  // the identity.
  //
  // The rotation issue can be solved by making the Z matrix lower triangular
  // (e.g. zero above the diagonal).  Z is tall and skinny, becasue there are
  // more time series than factors.  Making it lower triangular means the first
  // time series is only affected by the first factor.  The second is affected
  // by the first two factors, etc.  This is uncompelling, because if the first
  // 4 series are driven by one factor, and the next 3 by another, this
  // structure gets lost in the identifiability constraint.  Hmmmm... but the
  // coefficients can be zero...., so let's give this a try.
  //
  // The prior on the coefficients is a row-wise spike and slab prior.  The
  // spikes are modified to enforce zeros on the

  class GeneralSharedLocalLevelPosteriorSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model: The shared local level state model to be sampled.
    //   innovation_precision_priors: Independent prior distributions for the
    //     precisions of the random walk innovations.  One prior is needed for
    //     each random factor in 'model.'
    //   coefficient_prior_mean: A 'ydim x nfactors' matrix containing the prior
    //     mean of the observation coefficients.
    //   seeding_rng: The random number generator used to seed the RNG for this
    //     sampler.
    GeneralSharedLocalLevelPosteriorSampler(
        GeneralSharedLocalLevelStateModel *model,
        const std::vector<Ptr<MvnBase>> &slabs,
        const std::vector<Ptr<VariableSelectionPrior>> &spikes,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    void limit_model_selection(int max_flips);

   private:
    void draw_inclusion_indicators(int which_series);
    void draw_coefficients_given_inclusion(int which_series);

    GeneralSharedLocalLevelStateModel *model_;

    // The spikes_ and slabs_ vectors each have one element per time series.
    // These are the models included in samplers_.
    std::vector<Ptr<MvnBase>> slabs_;
    std::vector<Ptr<VariableSelectionPrior>> spikes_;

    // In most Glm's the coefficients are GlmCoefs, which carry their own
    // Selector objects indicating which coefficients are included.  In this
    // case the coefficients are rows or columns in a matrix, so the inclusion
    // indicators have to be stored externally.
    std::vector<Selector> inclusion_indicators_;

    // There is one element of samplers_ for each time series.
    std::vector<SpikeSlabSampler> samplers_;
  };

  class ConditionallyIndependentSharedLocalLevelPosteriorSampler
      : public PosteriorSampler {
   public:
    // Args:
    //   model:  The model to be managed.
    //   slabs: Prior distribution on the conditionally nonzero part of the
    //     observation coefficients.  One element for each potentially observed
    //     time series.  Dimension of each element is the number of factors.
    //   spikes: Prior distribution on the conditionally nonzero part of the
    //     observation coefficients.  One element for each potentially observed
    //     time series.
    //   sigsq: A vector of UnivParams pointers containing the residual
    //     variances from the host model.  If the host model is a mixture of
    //     Gaussians (e.g. probit, logit, T, Poisson, etc) then these "residual
    //     variances" are either dummy values (probably 1.0) or they contain
    //     pointers to the host model and can generate appropriate values on the
    //     fly.
    //   seeding_rng: The random number generator used to seed the RNG for the
    //     PosteriorSampler object.
    ConditionallyIndependentSharedLocalLevelPosteriorSampler(
        ConditionallyIndependentSharedLocalLevelStateModel *model,
        const std::vector<Ptr<MvnBase>> &slabs,
        const std::vector<Ptr<VariableSelectionPrior>> &spikes,
        const std::vector<Ptr<UnivParams>> &sigsq,
        RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;
    void limit_model_selection(int max_flips);

   private:
    void draw_inclusion_indicators(int which_series);
    void draw_coefficients_given_inclusion(int which_series);

    ConditionallyIndependentSharedLocalLevelStateModel *model_;

    // The spikes_ and slabs_ vectors each have one element per time series.
    // These are the models included in samplers_.
    std::vector<Ptr<MvnBase>> slabs_;
    std::vector<Ptr<VariableSelectionPrior>> spikes_;

    // There is one element of samplers_ for each time series.
    std::vector<SpikeSlabSampler> samplers_;

    std::vector<Ptr<UnivParams>> sigsq_;
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_SHARED_LOCAL_LEVEL_POSTERIOR_SAMPLER_HPP_

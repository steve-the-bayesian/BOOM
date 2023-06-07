#ifndef BOOM_STATE_SPACE_SHARED_SEASONAL_POSTERIOR_SAMPLER_HPP_
#define BOOM_STATE_SPACE_SHARED_SEASONAL_POSTERIOR_SAMPLER_HPP_
/*
  Copyright (C) 2023 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StateModels/SharedSeasonal.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/MvnBase.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"

namespace BOOM {

  // A posterior sampler for the shared seasonal state model.  The model assumes
  // that the k "factors" in the vector alpha are linearly related to m observed
  // series in y:
  //
  //          y[t] =    Z * alpha[t] + epsilon[t]
  //    alpha[t+1] = T[t] * alpha[t] + eta[t]
  //
  // The parameters in this model are Z, which contains a constrained set of
  // regression coefficients.  The variance of epsilon[t] is assumed to be a
  // normal mixture where the elements of epsilon[t] are conditionally
  // independent given a latent variable w[t] that has been imputed as part of a
  // previously run latent data imputation step.
  //
  // The prior assumed here is a product of independent regression priors on the
  // rows of Z, which is to say that each time series has it's own spike and
  // slab prior on the different state factors.
  //
  // See the comments to the SharedSeasonalStateModel for a full description of
  // the state, which includes current factor values and several lags.  The
  // "interesting" dimension of Z is equal to the number of factors, because
  // past seasonal lags don't contribute, so some elements of Z are structurally
  // zero.
  class SharedSeasonalPosteriorSampler
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
    SharedSeasonalPosteriorSampler(
        SharedSeasonalStateModel *model,
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

    SharedSeasonalStateModel *model_;

    // The spikes_ and slabs_ vectors each have one element per time series.
    // These are the models included in samplers_.
    std::vector<Ptr<MvnBase>> slabs_;
    std::vector<Ptr<VariableSelectionPrior>> spikes_;

    // There is one element of samplers_ for each time series.
    std::vector<SpikeSlabSampler> samplers_;

    std::vector<Ptr<UnivParams>> sigsq_;
  };

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_SHARED_SEASONAL_POSTERIOR_SAMPLER_HPP_

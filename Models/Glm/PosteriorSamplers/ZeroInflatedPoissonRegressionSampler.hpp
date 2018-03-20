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
#ifndef BOOM_ZERO_INFLATED_POISSON_REGRESSION_SAMPLER_HPP_
#define BOOM_ZERO_INFLATED_POISSON_REGRESSION_SAMPLER_HPP_

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/Glm/PoissonRegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitCompositeSpikeSlabSampler.hpp"
#include "Models/Glm/PosteriorSamplers/PoissonRegressionSpikeSlabSampler.hpp"
#include "Models/Glm/ZeroInflatedPoissonRegression.hpp"

namespace BOOM {

  class ZeroInflatedPoissonRegressionSampler : public PosteriorSampler {
   public:
    ZeroInflatedPoissonRegressionSampler(
        ZeroInflatedPoissonRegressionModel *model,
        const Ptr<VariableSelectionPrior> &poisson_spike,
        const Ptr<MvnBase> &poisson_slab,
        const Ptr<VariableSelectionPrior> &logit_spike,
        const Ptr<MvnBase> &logit_slab, RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    // Augment the latent data corresponding to the "forced zero"
    // indicators.
    // Args:
    //   stochastic: If 'true' then draw the indicators from their
    //     posterior distribution given observed data and model
    //     parameters.  If 'false' then replace each indicator with
    //     its posterior mean, which is the E step in an EM algorithm
    //     for finding the posterior mode.
    void impute_forced_zeros(bool stochastic);

    // Model selection is on by default.  allow_model_selection(false)
    // turns it off.  allow_model_selection(true) turns it back on
    // again.
    void allow_model_selection(bool tf);

    // Sets the parameters of the managed model to their posterior
    // modal values.  If the return value is true then the algorithm
    // converged.  If not, then it did not converge within a specified
    // number of iterations.
    void find_posterior_mode(double epsilon = 1e-5) override;
    bool posterior_mode_found() const { return posterior_mode_found_; }

   private:
    // Check that the latent models have latent data assigned to them,
    // that it is of the appropriate size, and that it matches a small
    // random sample of data from the actual model.  If it does not
    // match, then refresh_latent_data() is called to clear the latent
    // data and provide a fresh set.
    void ensure_latent_data();

    // Clear the data from poisson_ and logit_, create new data and
    // use it to populate the models.
    void refresh_latent_data();

    // Compute a measure of how much the coefficients have changed
    // from the last iteration of an iterative optimization algorithm.
    double compute_convergence_criterion(
        const Vector &old_logit_coefficients,
        const Vector &old_poisson_coefficients) const;

    ZeroInflatedPoissonRegressionModel *model_;

    Ptr<PoissonRegressionModel> poisson_;
    Ptr<BinomialLogitModel> logit_;
    Ptr<PoissonRegressionSpikeSlabSampler> poisson_sampler_;
    Ptr<BinomialLogitCompositeSpikeSlabSampler> logit_sampler_;

    bool posterior_mode_found_;
  };

}  // namespace BOOM

#endif  // BOOM_ZERO_INFLATED_POISSON_REGRESSION_SAMPLER_HPP_

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

#ifndef BOOM_QUANTILE_REGRESSION_POSTERIOR_SAMPLER_HPP_
#define BOOM_QUANTILE_REGRESSION_POSTERIOR_SAMPLER_HPP_

#include "Models/Glm/PosteriorSamplers/QuantileRegressionPosteriorSampler.hpp"
#include "Models/Glm/PosteriorSamplers/SpikeSlabSampler.hpp"
#include "Models/Glm/QuantileRegressionModel.hpp"
#include "Models/Glm/WeightedRegressionModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/Imputer.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  // The data augmentation scheme for quantile regression goes
  // something like this.  The quantile regression model is to
  // minimize the objective function \sum \rho_tau (y - x.dot(beta)),
  // where rho(u) = tau * u if u > 0 and rho(u) = -(1-tau) * u if u <
  // 0.  There are several ways to write rho in terms of absolute
  // value.  One is 2*rho(u) = fabs(u) + (2*tau - 1)*u.
  //
  // We can write exp(-2 * rho(u)) = \int f(u, lambda)*f(lambda) dlambda,
  // where
  //
  // f(u, lambda) = Normal(u | lambda * (2*tau - 1), lambda)
  // f(lambda) = Exponential (2 * tau * (1 - tau))  (or maybe its inverse).
  //
  // Now substitute Y - X * beta for u.
  //
  //--------------------------------------------
  // Full conditional distributions
  //--------------------------------------------
  // Given y, x, and beta, the full conditional distribution for
  // lambda[i] is that 1.0 / lambda[i] follows an inverse Gaussian
  // distribution with mean mu = 1.0 / fabs(r[i]), and shape parameter
  // 1.
  //
  // Given lambda[i], subtract the offset lambda * (2*tau - 1) from
  // y[i], and based on the adjusted y you've got a weighted least
  // squares regression with weightes 1.0 / lambda[i].

  //======================================================================
  // An imputation worker for simulating the complete data in a
  // quantile regression problem.
  class QuantileRegressionImputeWorker
      : public SufstatImputeWorker<RegressionData, WeightedRegSuf> {
   public:
    QuantileRegressionImputeWorker(const GlmCoefs *coefficients,
                                   double quantile, WeightedRegSuf &global_suf,
                                   std::mutex &global_suf_mutex,
                                   RNG *rng = nullptr,
                                   RNG &seeding_rng = GlobalRng::rng)
        : SufstatImputeWorker<RegressionData, WeightedRegSuf>(
              global_suf, global_suf_mutex, rng, seeding_rng),
          coefficients_(coefficients),
          quantile_complement_(1 - quantile) {}

    double adjusted_observation(double y, double lambda) const {
      return y - (2 * quantile_complement_ - 1) * lambda;
    }

    void impute_latent_data_point(const RegressionData &data_point,
                                  WeightedRegSuf *suf, RNG &rng) override;

   private:
    const GlmCoefs *coefficients_;
    double quantile_complement_;
  };

  //======================================================================
  // A quantile regression sampler for use with a fixed set of
  // predictors.
  class QuantileRegressionPosteriorSampler
      : public PosteriorSampler,
        public LatentDataSampler<QuantileRegressionImputeWorker> {
   public:
    QuantileRegressionPosteriorSampler(QuantileRegressionModel *model,
                                       const Ptr<MvnBase> &prior,
                                       RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;
    void draw_params();
    Ptr<QuantileRegressionImputeWorker> create_worker(std::mutex &m) override;
    void clear_latent_data() override;
    void assign_data_to_workers() override;
    const WeightedRegSuf &suf() const { return suf_; }

   private:
    QuantileRegressionModel *model_;
    Ptr<MvnBase> prior_;
    WeightedRegSuf suf_;
  };

  //======================================================================
  // A quantile regression sampler for use when model selection is
  // desired.
  class QuantileRegressionSpikeSlabSampler
      : public QuantileRegressionPosteriorSampler {
   public:
    QuantileRegressionSpikeSlabSampler(QuantileRegressionModel *model,
                                       const Ptr<MvnBase> &slab,
                                       const Ptr<VariableSelectionPrior> &spike,
                                       RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;

    // If allow == true then draw_model_indicators is a no-op.  Otherwise
    // model indicators will be sampled each iteration.
    void allow_model_selection(bool allow) {
      sam_.allow_model_selection(allow);
    }

    // In very large problems you may not want to sample every element
    // of the inclusion vector each time.  If max_flips is set to a
    // positive number then at most that many randomly chosen
    // inclusion indicators will be sampled.
    void limit_model_selection(int max_flips) {
      sam_.limit_model_selection(max_flips);
    }

   private:
    SpikeSlabSampler sam_;
    Ptr<MvnBase> slab_prior_;
    Ptr<VariableSelectionPrior> spike_prior_;
  };

}  // namespace BOOM
#endif  // BOOM_QUANTILE_REGRESSION_POSTERIOR_SAMPLER_HPP_

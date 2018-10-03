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

#include "Models/Glm/PosteriorSamplers/RegressionShrinkageSampler.hpp"
#include "LinAlg/Cholesky.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef RegressionShrinkageSampler::CoefficientGroup CG;
  }

  CG::CoefficientGroup(const Ptr<GaussianModelBase> &prior,
                       const std::vector<int> &indices)
      : prior_(prior), indices_(indices) {}

  void CG::refresh_sufficient_statistics(const Vector &beta) {
    prior_->suf()->clear();
    for (int i = 0; i < indices_.size(); ++i) {
      prior_->suf()->update_raw(beta[indices_[i]]);
    }
  }

  //===========================================================================
  RegressionShrinkageSampler::RegressionShrinkageSampler(
      RegressionModel *model,
      const Ptr<GammaModelBase> &residual_precision_prior,
      const std::vector<CoefficientGroup> &groups, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        variance_sampler_(residual_precision_prior),
        groups_(groups) {}

  void RegressionShrinkageSampler::draw() {
    draw_coefficients();
    draw_residual_variance();
    draw_hyperparameters();
  }

  void RegressionShrinkageSampler::draw_coefficients() {
    Vector prior_precision_diagonal = this->prior_precision_diagonal();
    SpdMatrix posterior_precision = model_->suf()->xtx() / model_->sigsq();
    posterior_precision.diag() += prior_precision_diagonal;

    Vector scaled_posterior_mean = model_->suf()->xty() / model_->sigsq();
    scaled_posterior_mean += prior_precision_diagonal * prior_mean();

    Cholesky cholesky(posterior_precision);
    Vector posterior_mean = cholesky.solve(scaled_posterior_mean);

    model_->set_Beta(rmvn_precision_upper_cholesky_mt(rng(), posterior_mean,
                                                      cholesky.getLT()));
  }

  void RegressionShrinkageSampler::draw_hyperparameters() {
    for (int g = 0; g < groups_.size(); ++g) {
      groups_[g].refresh_sufficient_statistics(model_->Beta());
      groups_[g].sample_posterior();
    }
  }

  void RegressionShrinkageSampler::draw_residual_variance() {
    double data_sum_of_squares = model_->suf()->relative_sse(model_->coef());
    double data_df = model_->suf()->n();
    double sigsq = variance_sampler_.draw(rng(), data_df, data_sum_of_squares);
    model_->set_sigsq(sigsq);
  }

  double RegressionShrinkageSampler::logpri() const {
    double ans = variance_sampler_.log_prior(model_->sigsq());
    const Vector &coefficients(model_->Beta());
    for (int g = 0; g < groups_.size(); ++g) {
      const std::vector<int> &indices(groups_[g].indices());
      for (int i = 0; i < indices.size(); ++i) {
        ans += groups_[g].log_prior(coefficients[indices[i]]);
      }
      ans += groups_[g].log_hyperprior();
    }
    return ans;
  }

  Vector RegressionShrinkageSampler::prior_mean() const {
    Vector ans(model_->xdim());
    for (int g = 0; g < groups_.size(); ++g) {
      double mean = groups_[g].prior_mean();
      const std::vector<int> &indices(groups_[g].indices());
      for (int i = 0; i < indices.size(); ++i) {
        ans[indices[i]] = mean;
      }
    }
    return ans;
  }

  Vector RegressionShrinkageSampler::prior_precision_diagonal() const {
    Vector ans(model_->xdim());
    for (int g = 0; g < groups_.size(); ++g) {
      double precision = 1.0 / groups_[g].prior_variance();
      const std::vector<int> &indices(groups_[g].indices());
      for (int i = 0; i < indices.size(); ++i) {
        ans[indices[i]] = precision;
      }
    }
    return ans;
  }

}  // namespace BOOM

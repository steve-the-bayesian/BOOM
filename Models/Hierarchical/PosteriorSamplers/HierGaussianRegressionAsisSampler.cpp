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

#include "Models/Hierarchical/PosteriorSamplers/HierGaussianRegressionAsisSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionCoefficientSampler.hpp"
#include "Models/PosteriorSamplers/MvnMeanSampler.hpp"
#include "Models/PosteriorSamplers/MvnVarSampler.hpp"

namespace BOOM {
  namespace {
    typedef HierGaussianRegressionAsisSampler HGRAS;
  }
  HGRAS::HierGaussianRegressionAsisSampler(
      HierarchicalGaussianRegressionModel *model,
      const Ptr<MvnModel> &coefficient_mean_hyperprior,
      const Ptr<WishartModel> &coefficient_precision_hyperprior,
      const Ptr<GammaModelBase> &residual_precision_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        coefficient_mean_hyperprior_(coefficient_mean_hyperprior),
        coefficient_precision_hyperprior_(coefficient_precision_hyperprior),
        residual_variance_prior_(residual_precision_prior),
        residual_variance_sampler_(residual_variance_prior_) {
    NEW(MvnMeanSampler, mean_sampler)
    (model_->prior(), coefficient_mean_hyperprior);
    model_->prior()->set_method(mean_sampler);
    NEW(MvnVarSampler, var_sampler)
    (model_->prior(), coefficient_precision_hyperprior);
    model_->prior()->set_method(var_sampler);
  }

  void HGRAS::draw() {
    MvnModel *prior = model_->prior();
    prior->clear_data();
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      RegressionModel *reg = model_->data_model(i);
      // Sample coefficients for model i.
      RegressionCoefficientSampler::sample_regression_coefficients(
          rng(), reg, *prior);
      prior->suf()->update_raw(reg->Beta());
    }
    prior->sample_posterior();

    if (xtx_.nrow() != prior->dim()) {
      refresh_working_suf();
    }
    xty_ = 0;
    Matrix centered_regression_effects(xty_.size(), model_->number_of_groups());
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      RegressionModel *reg = model_->data_model(i);
      // Convert coefficients to effects centered
      centered_regression_effects.col(i) = reg->Beta() - model_->prior()->mu();

      // Accumulate sufficient statistics to be used in the draw of the prior
      // mean.
      xty_ += reg->suf()->xty() -
              reg->suf()->xtx() * centered_regression_effects.col(i);
    }

    // With the regression effects centered, the prior mean is no longer viewed
    // as the mean of a bunch of MVN data, but as the set of regression
    // coefficients on a bunch of ajusted observations.
    prior->set_mu(RegressionCoefficientSampler::sample_regression_coefficients(
        rng(), xtx_, xty_, model_->residual_variance(),
        *coefficient_mean_hyperprior_));
    prior->set_siginv(MvnVarSampler::draw_precision(
        rng(), model_->number_of_groups(), centered_regression_effects.outer(),
        *coefficient_precision_hyperprior_));

    if (!!residual_variance_prior_) {
      // Convert centered_regression_effects betas back to betas, using the
      // newly drawn prior mean.  Accumulate the sufficient statistics needed to
      // draw the residual_variance.
      double sample_size = 0;
      double residual_sum_of_squares = 0;
      const Vector &prior_mean(model_->prior()->mu());
      for (int i = 0; i < model_->number_of_groups(); ++i) {
        RegressionModel *reg = model_->data_model(i);
        reg->set_Beta(prior_mean + centered_regression_effects.col(i));
        sample_size += reg->suf()->n();
        residual_sum_of_squares += reg->suf()->relative_sse(reg->coef());
      }
      // draw sigsq
      model_->set_residual_variance(residual_variance_sampler_.draw(
          rng(), sample_size, residual_sum_of_squares));
    }
  }

  double HGRAS::logpri() const {
    const MvnModel *prior = model_->prior();
    double ans = 0;
    if (!!residual_variance_prior_) {
      ans += residual_variance_sampler_.log_prior(model_->residual_variance());
    }
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      ans += prior->logp(model_->data_model(i)->Beta());
    }

    // Adjust this for the specific hyperprior used in the prior sampling given
    // beta residuals.
    ans += prior->logpri();
    return ans;
  }

  void HGRAS::set_hyperprior(
      const Ptr<MvnModel> &coefficient_mean_hyperprior,
      const Ptr<WishartModel> &coefficient_precision_hyperprior,
      const Ptr<GammaModelBase> &residual_precision_prior) {
    coefficient_mean_hyperprior_ = coefficient_mean_hyperprior;
    coefficient_precision_hyperprior_ = coefficient_precision_hyperprior;
    residual_variance_prior_ = residual_precision_prior;
    residual_variance_sampler_.set_prior(residual_variance_prior_);
  }

  void HGRAS::refresh_working_suf() {
    int xdim = model_->xdim();
    xtx_.resize(xdim);
    xty_.resize(xdim);
    xtx_ = 0;
    xty_ = 0;
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      xtx_ += model_->data_model(i)->suf()->xtx();
    }
  }

}  // namespace BOOM

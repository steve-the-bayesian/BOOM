// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Bart/PosteriorSamplers/GaussianLinearBartPosteriorSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  GaussianLinearBartPosteriorSampler::GaussianLinearBartPosteriorSampler(
      GaussianLinearBartModel *model,
      const ZellnerPriorParameters &regression_prior,
      const BartPriorParameters &bart_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        first_time_for_regression_(true),
        bart_sampler_(new GaussianBartPosteriorSampler(
            model->bart(), regression_prior.prior_sigma_guess,
            regression_prior.prior_sigma_guess_weight,
            bart_prior.total_prediction_sd, bart_prior.prior_tree_depth_alpha,
            bart_prior.prior_tree_depth_beta,
            PointMassPrior(model->bart()->number_of_trees()), seeding_rng)),
        first_time_for_bart_(true) {
    RegressionModel *regression = model_->regression();
    NEW(BregVsSampler, regression_sampler)(regression, regression_prior);
    regression->set_method(regression_sampler);

    model_->bart()->set_method(bart_sampler_);
  }

  double GaussianLinearBartPosteriorSampler::logpri() const {
    return model_->regression()->logpri() + bart_sampler_->logpri();
  }

  void GaussianLinearBartPosteriorSampler::draw() {
    sample_regression_posterior();
    sample_bart_posterior();
  }

  void GaussianLinearBartPosteriorSampler::sample_regression_posterior() {
    adjust_regression_residuals();
    model_->regression()->sample_posterior();
    model_->bart()->set_sigsq(model_->regression()->sigsq());
  }

  void GaussianLinearBartPosteriorSampler::sample_bart_posterior() {
    adjust_bart_residuals();
    // Calling the base method so that sigsq is not sampled again.
    bart_sampler_->BartPosteriorSamplerBase::draw();
    // Set the sigsq parameter in the bart model just in case it
    // changed for some reason.
    model_->bart()->set_sigsq(model_->regression()->sigsq());
  }

  void GaussianLinearBartPosteriorSampler::adjust_regression_residuals() {
    RegressionModel *regression = model_->regression();
    if (first_time_for_regression_) {
      regression->only_keep_sufstats(true);
      regression->use_normal_equations();
    }
    const GaussianBartModel *bart = model_->bart();
    regression->clear_data();
    const std::vector<Ptr<RegressionData> > &data(model_->dat());
    for (int i = 0; i < data.size(); ++i) {
      const RegressionData *dp(data[i].get());
      const Vector &x(dp->x());
      double y = dp->y();
      double residual = y - bart->predict(x);
      regression->suf()->add_mixture_data(residual, x, 1.0);
    }
    if (first_time_for_regression_) {
      regression->suf().dcast<NeRegSuf>()->fix_xtx(true);
      first_time_for_regression_ = false;
    }
  }

  void GaussianLinearBartPosteriorSampler::adjust_bart_residuals() {
    const RegressionModel *regression = model_->regression();
    GaussianBartModel *bart = model_->bart();
    if (first_time_for_bart_) {
      bart->finalize_data();
      bart_sampler_->check_residuals();
      first_time_for_bart_ = false;
    }
    const std::vector<Ptr<RegressionData> > &data(model_->dat());
    for (int i = 0; i < data.size(); ++i) {
      const RegressionData *dp(data[i].get());
      double residual =
          dp->y() - regression->predict(dp->x()) - bart->predict(dp->x());
      bart_sampler_->set_residual(i, residual);
    }
  }

}  // namespace BOOM

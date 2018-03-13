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

#include "Models/Hierarchical/PosteriorSamplers/HierarchicalGaussianRegressionSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionCoefficientSampler.hpp"

namespace BOOM {
  namespace {
    typedef HierarchicalGaussianRegressionSampler HGRS;
    typedef HierarchicalGaussianRegressionModel HGRM;
  }  // namespace

  HGRS::HierarchicalGaussianRegressionSampler(
      HGRM *model, const Ptr<GammaModelBase> &residual_precision_prior,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        residual_variance_prior_(residual_precision_prior),
        residual_variance_sampler_(residual_variance_prior_) {}

  // TODO:  Consider threads here.
  void HGRS::draw() {
    double sample_size = 0;
    double residual_sum_of_squares = 0;
    MvnModel *prior = model_->prior();
    prior->clear_data();
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      RegressionModel *reg = model_->data_model(i);
      RegressionCoefficientSampler::sample_regression_coefficients(rng(), reg,
                                                                   *prior);
      prior->suf()->update_raw(reg->Beta());
      sample_size += reg->suf()->n();
      residual_sum_of_squares += reg->suf()->relative_sse(reg->coef());
    }
    model_->set_residual_variance(residual_variance_sampler_.draw(
        rng(), sample_size, residual_sum_of_squares));
    prior->sample_posterior();
  }

  double HGRS::logpri() const {
    const MvnModel *prior = model_->prior();
    double ans =
        residual_variance_sampler_.log_prior(model_->residual_variance());
    for (int i = 0; i < model_->number_of_groups(); ++i) {
      ans += prior->logp(model_->data_model(i)->Beta());
    }
    ans += prior->logpri();
    return ans;
  }

}  // namespace BOOM

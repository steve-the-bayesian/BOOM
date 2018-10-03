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

#include "Models/Glm/PosteriorSamplers/RegressionCoefficientSampler.hpp"
#include "LinAlg/Cholesky.hpp"
#include "distributions.hpp"

namespace BOOM {

  RegressionCoefficientSampler::RegressionCoefficientSampler(
      RegressionModel *model, const Ptr<MvnBase> &prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), prior_(prior) {}

  void RegressionCoefficientSampler::draw() {
    sample_regression_coefficients(rng(), model_, *prior_);
  }

  void RegressionCoefficientSampler::sample_regression_coefficients(
      RNG &rng, RegressionModel *model, const MvnBase &prior) {
    SpdMatrix prior_precision = prior.siginv();
    SpdMatrix posterior_precision =
        model->suf()->xtx() / model->sigsq() + prior_precision;
    Vector scaled_posterior_mean = model->suf()->xty() / model->sigsq();
    scaled_posterior_mean += prior_precision * prior.mu();

    Cholesky cholesky(posterior_precision);
    Vector posterior_mean = cholesky.solve(scaled_posterior_mean);
    model->set_Beta(rmvn_precision_upper_cholesky_mt(rng, posterior_mean,
                                                     cholesky.getLT()));
  }

  Vector RegressionCoefficientSampler::sample_regression_coefficients(
      RNG &rng, const SpdMatrix &xtx, const Vector &xty, double sigsq,
      const MvnBase &prior) {
    SpdMatrix prior_precision = prior.siginv();
    SpdMatrix posterior_precision = (xtx / sigsq) + prior_precision;
    Vector scaled_posterior_mean = xty / sigsq + prior_precision * prior.mu();
    Cholesky cholesky(posterior_precision);
    Vector posterior_mean = cholesky.solve(scaled_posterior_mean);
    return rmvn_precision_upper_cholesky_mt(rng, posterior_mean,
                                            cholesky.getLT());
  }

  double RegressionCoefficientSampler::logpri() const {
    return prior_->logp(model_->Beta());
  }

}  // namespace BOOM

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

#include "Models/PosteriorSamplers/GaussianGivenSigmaSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  GaussianGivenSigmaSampler::GaussianGivenSigmaSampler(
      GaussianModelGivenSigma *model, const Ptr<GaussianModelBase> &mean_prior,
      const Ptr<GammaModelBase> &sample_size_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        mean_prior_(mean_prior),
        sample_size_prior_(sample_size_prior) {}

  void GaussianGivenSigmaSampler::draw() {
    draw_mean();
    draw_sample_size();
  }

  double GaussianGivenSigmaSampler::logpri() const {
    return mean_prior_->logp(model_->mu()) +
           sample_size_prior_->logp(model_->kappa());
  }

  void GaussianGivenSigmaSampler::draw_mean() {
    Ptr<GaussianSuf> suf = model_->suf();
    double posterior_precision =
        (1.0 / mean_prior_->sigsq()) + (suf->n() / model_->sigsq());
    double scaled_posterior_mean = (suf->sum() / model_->sigsq()) +
                                   (mean_prior_->mu() / mean_prior_->sigsq());
    double posterior_mean = scaled_posterior_mean / posterior_precision;
    double posterior_sd = 1.0 / sqrt(posterior_precision);
    model_->set_mu(rnorm_mt(rng(), posterior_mean, posterior_sd));
  }

  void GaussianGivenSigmaSampler::draw_sample_size() {
    Ptr<GaussianSuf> suf = model_->suf();
    double a = sample_size_prior_->alpha() + .5 * suf->n();
    double b =
        sample_size_prior_->beta() +
        .5 * suf->centered_sumsq(model_->mu()) / model_->scaling_variance();
    model_->set_kappa(rgamma_mt(rng(), a, b));
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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
#include "Models/PosteriorSamplers/ZeroInflatedPoissonSampler.hpp"
#include "Models/ZeroInflatedPoissonModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  ZeroInflatedPoissonSampler::ZeroInflatedPoissonSampler(
      ZeroInflatedPoissonModel *model, const Ptr<GammaModel> &lambda_prior,
      const Ptr<BetaModel> &zero_prob_prior, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        lambda_prior_(lambda_prior),
        zero_probability_prior_(zero_prob_prior) {}

  ZeroInflatedPoissonSampler *ZeroInflatedPoissonSampler::clone_to_new_host(
      Model *new_host) const {
    return new ZeroInflatedPoissonSampler(
        dynamic_cast<ZeroInflatedPoissonModel *>(new_host),
        lambda_prior_->clone(),
        zero_probability_prior_->clone(),
        rng());
  }

  void ZeroInflatedPoissonSampler::draw() {
    double p = model_->zero_probability();
    double pbinomial = p;
    double ppoisson = (1 - p) * dpois(0, model_->lambda());
    double nc = pbinomial + ppoisson;
    pbinomial /= nc;

    int nzero = lround(model_->suf()->number_of_zeros());

    double nzero_binomial = rbinom_mt(rng(), nzero, pbinomial);
    double nzero_poission = nzero - nzero_binomial;

    int counter = 0;
    do {
      if (++counter > 1000) {
        report_error("rbeta produced the value 0 over 1000 times.");
      }
      p = rbeta_mt(rng(), zero_probability_prior_->a() + nzero_binomial,
                   zero_probability_prior_->b() + nzero - nzero_binomial +
                       model_->suf()->number_of_positives());
    } while (p <= 0.0 || p >= 1.0);
    model_->set_zero_probability(p);

    double a = lambda_prior_->alpha() + model_->suf()->sum_of_positives();
    double b = lambda_prior_->beta() + model_->suf()->number_of_positives();
    b += nzero_poission;
    double lambda = -1;  // need to declare lambda before the do loop.
    do {
      if (++counter > 1000) {
        report_error("rgamma produced the value 0 over 1000 times.");
      }
      lambda = rgamma_mt(rng(), a, b);
    } while (lambda <= 0.0);
    model_->set_lambda(lambda);
  }

  double ZeroInflatedPoissonSampler::logpri() const {
    double ans = zero_probability_prior_->logp(model_->zero_probability());
    ans += lambda_prior_->logp(model_->lambda());
    return ans;
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/MLVS_data_imputer.hpp"

#include "cpputil/lse.hpp"
#include "cpputil/math_utils.hpp"

#include "stats/logit.hpp"

#include <cmath>
#include "distributions.hpp"

namespace BOOM {

  MlvsDataImputer::MlvsDataImputer(SufficientStatistics &global_suf,
                                   std::mutex &global_suf_mutex,
                                   MultinomialLogitModel *model, RNG *rng,
                                   RNG &seeding_rng)
      : SufstatImputeWorker<ChoiceData, SufficientStatistics>(
            global_suf, global_suf_mutex, rng, seeding_rng),
        model_(model),
        mu_(Vector{5.09, 3.29, 1.82, 1.24, 0.76, 0.39, 0.04, -0.31, -0.67, -1.06}),
        sigsq_inv_(pow(Vector({4.5, 2.02, 1.1, 0.42, 0.2, 0.11, 0.08, 0.08, 0.09, 0.15}), -1)),
        sd_(pow(sigsq_inv_, -0.5)),
        log_mixing_weights_(log(Vector({
                0.004, 0.04, 0.168, 0.147, 0.125, 0.101, 0.104, 0.116, 0.107, 0.088}))),
        log_sampling_probs_(model_->log_sampling_probs()),
        downsampling_(log_sampling_probs_.size() == model_->Nchoices()),
        post_prob_(log_mixing_weights_),
        u(model_->Nchoices()),
        eta(u),
        wgts(u) {}

  void MlvsDataImputer::impute_latent_data_point(const ChoiceData &dp,
                                                 SufficientStatistics *suf,
                                                 RNG &rng) {
    model_->fill_eta(dp, eta);  // eta+= downsampling_logprob
    if (downsampling_) eta += log_sampling_probs_;  //
    uint M = model_->Nchoices();
    uint y = dp.value();
    assert(y < M);
    double loglam = lse(eta);
    double logzmin = rlexp_mt(rng, loglam);
    u[y] = -logzmin;
    for (uint m = 0; m < M; ++m) {
      if (m != y) {
        double tmp = rlexp_mt(rng, eta[m]);
        double logz = lse2(logzmin, tmp);
        u[m] = -logz;
      }
      uint k = unmix(rng, u[m] - eta[m]);
      u[m] -= mu_[k];
      wgts[m] = sigsq_inv_[k];
    }
    suf->update(dp, wgts, u);
  }

  //----------------------------------------------------------------------
  uint MlvsDataImputer::unmix(RNG &rng, double u) const {
    uint K = post_prob_.size();
    for (uint k = 0; k < K; ++k)
      post_prob_[k] = log_mixing_weights_[k] + dnorm(u, mu_[k], sd_[k], true);
    post_prob_.normalize_logprob();
    return rmulti_mt(rng, post_prob_);
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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
#include "Models/PosteriorSamplers/PoissonGammaSampler.hpp"
#include "Models/GammaModel.hpp"
#include "Models/PoissonModel.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  PoissonGammaSampler::PoissonGammaSampler(PoissonModel *model,
                                           const Ptr<GammaModel> &prior,
                                           RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_(prior)
  {}

  PoissonGammaSampler *PoissonGammaSampler::clone_to_new_host(
      Model *new_host) const {
    return new PoissonGammaSampler(
        dynamic_cast<PoissonModel *>(new_host),
        prior_->clone(),
        rng());
  }

  double PoissonGammaSampler::alpha() const { return prior_->alpha(); }

  double PoissonGammaSampler::beta() const { return prior_->beta(); }

  double PoissonGammaSampler::logpri() const {
    return dgamma(model_->lam(), alpha(), beta(), true);
  }

  void PoissonGammaSampler::draw() {
    double n = model_->suf()->n();
    double sum = model_->suf()->sum();
    double a = sum + prior_->alpha();
    double b = n + prior_->beta();
    int number_of_attempts = 0;
    double lambda;
    do {
      if (++number_of_attempts > 100) {
        report_error(
            "Too many attempts trying to draw lambda in "
            "PoissonGammaSampler::draw.");
      }
      lambda = rgamma_mt(rng(), a, b);
    } while (!std::isfinite(lambda) || lambda <= 0.0);
    model_->set_lam(lambda);
  }

  void PoissonGammaSampler::find_posterior_mode(double) {
    double n = model_->suf()->n();
    double sum = model_->suf()->sum();
    double a = sum + prior_->alpha();
    double b = n + prior_->beta();
    double mode = (a - 1) / b;
    if (mode < 0) mode = 0;
    model_->set_lam(mode);
  }
}  // namespace BOOM

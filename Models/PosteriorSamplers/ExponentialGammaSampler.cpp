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
#include "Models/PosteriorSamplers/ExponentialGammaSampler.hpp"
#include "Models/ExponentialModel.hpp"
#include "Models/GammaModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef ExponentialGammaSampler EGS;

  EGS::ExponentialGammaSampler(ExponentialModel *model,
                               const Ptr<GammaModel> &prior,
                               RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_(prior) {}

  EGS *EGS::clone_to_new_host(Model *new_host) const {
    return new EGS(dynamic_cast<ExponentialModel *>(new_host),
                   prior_->clone(),
                   rng());
  }

  void EGS::draw() {
    double a = model_->suf()->n() + prior_->alpha();
    double b = model_->suf()->sum() + prior_->beta();
    model_->set_lam(rgamma_mt(rng(), a, b));
  }

  void EGS::find_posterior_mode(double) {
    double a = model_->suf()->n() + prior_->alpha();
    double b = model_->suf()->sum() + prior_->beta();
    double mode = (a - 1) / b;
    model_->set_lam(std::max<double>(mode, 0.0));
  }

  double EGS::logpri() const {
    double lam = model_->lam();
    return dgamma(lam, a(), b(), true);
  }

  double EGS::a() const { return prior_->alpha(); }
  double EGS::b() const { return prior_->beta(); }

}  // namespace BOOM

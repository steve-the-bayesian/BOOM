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
#include "Models/PosteriorSamplers/GaussianMeanSampler.hpp"
#include <cmath>
#include "Models/GaussianModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef GaussianMeanSampler GMS;

  GMS::GaussianMeanSampler(GaussianModel *model, double mu, double tau,
                           RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        prior_(new GaussianModel(mu, tau)) {}

  GMS::GaussianMeanSampler(GaussianModel *model, const Ptr<GaussianModel> &prior,
                           RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), model_(model), prior_(prior) {}

  GMS *GMS::clone_to_new_host(Model *new_host) const {
    return new GMS(dynamic_cast<GaussianModel *>(new_host),
                   prior_->clone(),
                   rng());
  }

  double GMS::logpri() const { return prior_->logp(model_->mu()); }

  void GMS::draw() {
    Ptr<GaussianSuf> s = model_->suf();

    double ybar = s->ybar();
    double n = s->n();

    double sigsq = model_->sigsq();

    double mu0 = prior_->mu();
    double tausq = prior_->sigsq();

    double ivar = (n / sigsq) + (1.0 / tausq);
    double mean = (n * ybar / sigsq + mu0 / tausq) / ivar;
    double sd = sqrt(1.0 / ivar);

    double ans = rnorm_mt(rng(), mean, sd);
    model_->set_mu(ans);
  }
}  // namespace BOOM

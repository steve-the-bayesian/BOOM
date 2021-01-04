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

#include "Models/Glm/PosteriorSamplers/MlogitRwm.hpp"
#include "Models/MvnModel.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef MultinomialLogitModel MLM;
  typedef MlogitRwm MLR;

  MLR::MlogitRwm(MLM *mlm, const Ptr<MvnBase> &pri, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mlm_(mlm), pri_(pri) {}

  MLR::MlogitRwm(MLM *mlm, const Vector &mu, const SpdMatrix &Ominv,
                 RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mlm_(mlm),
        pri_(new MvnModel(mu, Ominv, true)) {}

  void MLR::draw() {
    // random walk metropolis centered on current beta, with inverse
    // variance matrix given by current hessian of log posterior

    const Selector &inc(mlm_->coef().inc());
    uint p = inc.nvars();
    H.resize(p);
    g.resize(p);
    Vector nonzero_beta = mlm_->coef().included_coefficients();
    mu = inc.select(pri_->mu());
    ivar = inc.select(pri_->siginv());

    double logp_old = mlm_->Loglike(nonzero_beta, g, H, 2) +
                      dmvn(nonzero_beta, mu, ivar, 0, true);

    H *= -1;
    H += ivar;  // now H is inverse posterior variance
    bstar = rmvt_ivar(nonzero_beta, H, 3);

    double logp_new = mlm_->loglike(bstar) + dmvn(bstar, mu, ivar, 0, true);
    double log_alpha = logp_new - logp_old;
    double logu = log(runif_mt(rng(), 0, 1));
    while (!std::isfinite(logu)) logu = log(runif_mt(rng(), 0, 1));
    if (logu > log_alpha) {
      // Reject the draw.  Do nothing here.
    } else {  // Accept the draw
      mlm_->coef().set_included_coefficients(bstar);
    }
  }

  double MLR::logpri() const {
    const Selector &inc(mlm_->coef().inc());
    Vector b = mlm_->coef().included_coefficients();
    return dmvn(b, inc.select(pri_->mu()), inc.select(pri_->siginv()), true);
  }
}  // namespace BOOM

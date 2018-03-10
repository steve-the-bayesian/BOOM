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

  PoissonGammaSampler::PoissonGammaSampler(PoissonModel *p,
                                           const Ptr<GammaModel> &g,
                                           RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), pois(p), gam(g) {}

  double PoissonGammaSampler::alpha() const { return gam->alpha(); }

  double PoissonGammaSampler::beta() const { return gam->beta(); }

  double PoissonGammaSampler::logpri() const {
    double lam = pois->lam();
    return dgamma(lam, alpha(), beta(), true);
  }

  void PoissonGammaSampler::draw() {
    double n = pois->suf()->n();
    double sum = pois->suf()->sum();
    double a = sum + gam->alpha();
    double b = n + gam->beta();
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
    pois->set_lam(lambda);
  }

  void PoissonGammaSampler::find_posterior_mode(double) {
    double n = pois->suf()->n();
    double sum = pois->suf()->sum();
    double a = sum + gam->alpha();
    double b = n + gam->beta();
    double mode = (a - 1) / b;
    if (mode < 0) mode = 0;
    pois->set_lam(mode);
  }
}  // namespace BOOM

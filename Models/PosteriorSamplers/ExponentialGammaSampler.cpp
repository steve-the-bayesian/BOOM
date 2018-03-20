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

  EGS::ExponentialGammaSampler(ExponentialModel *Mod,
                               const Ptr<GammaModel> &Pri, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mod(Mod), pri(Pri) {}

  void EGS::draw() {
    double a = mod->suf()->n() + pri->alpha();
    double b = mod->suf()->sum() + pri->beta();
    mod->set_lam(rgamma_mt(rng(), a, b));
  }

  void EGS::find_posterior_mode(double) {
    double a = mod->suf()->n() + pri->alpha();
    double b = mod->suf()->sum() + pri->beta();
    double mode = (a - 1) / b;
    mod->set_lam(std::max<double>(mode, 0.0));
  }

  double EGS::logpri() const {
    double lam = mod->lam();
    return dgamma(lam, a(), b(), true);
  }

  double EGS::a() const { return pri->alpha(); }
  double EGS::b() const { return pri->beta(); }

}  // namespace BOOM

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
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"
#include "Models/DirichletModel.hpp"
#include "Models/MultinomialModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef MultinomialDirichletSampler MDS;
  typedef MultinomialModel MM;
  typedef DirichletModel DM;

  MDS::MultinomialDirichletSampler(MM *Mod, const Vector &Nu, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mod_(Mod), pri_(new DM(Nu)) {}

  MDS::MultinomialDirichletSampler(MM *Mod, const Ptr<DM> &prior,
                                   RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mod_(Mod), pri_(prior) {}

  MDS::MultinomialDirichletSampler(const MDS &rhs)
      : PosteriorSampler(rhs),
        mod_(rhs.mod_->clone()),
        pri_(rhs.pri_->clone()) {}

  MDS *MDS::clone() const { return new MDS(*this); }

  void MDS::draw() {
    Vector counts = pri_->nu() + mod_->suf()->n();
    Vector pi = rdirichlet_mt(rng(), counts);
    mod_->set_pi(pi);
  }

  void MDS::find_posterior_mode(double) {
    Vector counts = pri_->nu() + mod_->suf()->n();
    Vector pi = mdirichlet(counts);
    mod_->set_pi(pi);
  }

  double MDS::logpri() const { return pri_->logp(mod_->pi()); }

}  // namespace BOOM

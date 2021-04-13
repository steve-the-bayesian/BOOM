// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/PosteriorSamplers/MvnMeanSampler.hpp"
#include "Models/MvnModel.hpp"
#include "Models/ParamTypes.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
namespace BOOM {

  typedef MvnConjMeanSampler MCS;

  MCS::MvnConjMeanSampler(MvnModel *Mod, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mvn(Mod),
        mu0(new VectorParams(Mod->mu().zero())),
        kappa(new UnivParams(0.0)) {}

  MCS::MvnConjMeanSampler(MvnModel *Mod, const Ptr<VectorParams> &Mu0,
                          const Ptr<UnivParams> &Kappa, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mvn(Mod), mu0(Mu0), kappa(Kappa) {}

  MCS::MvnConjMeanSampler(MvnModel *Mod, const Vector &Mu0, double Kappa,
                          RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mvn(Mod),
        mu0(new VectorParams(Mu0)),
        kappa(new UnivParams(Kappa)) {}

  MCS * MCS::clone_to_new_host(Model *new_host) const {
    return new MCS(dynamic_cast<MvnModel *>(new_host),
                   mu0->clone(),
                   kappa->clone(),
                   rng());
  }

  void MCS::draw() {
    Ptr<MvnSuf> s = mvn->suf();
    double n = s->n();
    double k = kappa->value();
    const SpdMatrix &Siginv(mvn->siginv());
    SpdMatrix ivar = (n + k) * Siginv;
    double w = n / (n + k);
    Vector mu = w * s->ybar() + (1.0 - w) * mu0->value();
    mu = rmvn_ivar_mt(rng(), mu, ivar);
    mvn->set_mu(mu);
  }

  double MCS::logpri() const {
    double k = kappa->value();
    if (k == 0.0) return BOOM::negative_infinity();
    const Ptr<SpdParams> Sig = mvn->Sigma_prm();
    const Vector &mu(mvn->mu());
    uint d = mvn->dim();
    double ldsi = d * log(k) + Sig->ldsi();
    return dmvn(mu, mu0->value(), k * Sig->ivar(), ldsi, true);
  }

  //----------------------------------------------------------------------
  typedef MvnMeanSampler MMS;

  MMS::MvnMeanSampler(MvnModel *m, const Ptr<VectorParams> &Mu0,
                      const Ptr<SpdParams> &Omega, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mvn(m),
        mu_prior_(new MvnModel(Mu0, Omega)) {}

  MMS::MvnMeanSampler(MvnModel *m, const Ptr<MvnBase> &Pri, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), mvn(m), mu_prior_(Pri) {}

  MMS::MvnMeanSampler(MvnModel *m, const Vector &Mu0, const SpdMatrix &Omega,
                      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        mvn(m),
        mu_prior_(new MvnModel(Mu0, Omega)) {}

  MMS *MMS::clone_to_new_host(Model *new_host) const {
    return new MMS(
        dynamic_cast<MvnModel *>(new_host),
        mu_prior_->clone(),
        rng());
  }

  double MMS::logpri() const { return mu_prior_->logp(mvn->mu()); }

  void MMS::draw() {
    Ptr<MvnSuf> s = mvn->suf();
    double n = s->n();
    const SpdMatrix &siginv(mvn->siginv());
    const SpdMatrix &ominv(mu_prior_->siginv());
    SpdMatrix Ivar = n * siginv + ominv;
    Vector mu = Ivar.solve(n * (siginv * s->ybar()) + ominv * mu_prior_->mu());
    mu = rmvn_ivar(mu, Ivar);
    mvn->set_mu(mu);
  }
}  // namespace BOOM

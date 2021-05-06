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
#ifndef BOOM_MVN_MEAN_SAMPLER_HPP
#define BOOM_MVN_MEAN_SAMPLER_HPP
#include "Models/MvnBase.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class MvnModel;
  class VectorParams;
  class SpdParams;

  //____________________________________________________________
  class MvnConjMeanSampler : public PosteriorSampler {
    // assumes y~N(mu, Sig) with mu~N(mu0, Sig/kappa)
    // draws mu given y, Sigma, mu0, kappa
   public:
    explicit MvnConjMeanSampler(MvnModel *Mod,  // improper: mu0 = 0 kappa = 0;
                                RNG &seeding_rng = GlobalRng::rng);
    MvnConjMeanSampler(MvnModel *Mod,
                       const Ptr<VectorParams> &Mu0,
                       const Ptr<UnivParams> &Kappa,
                       RNG &seeding_rng = GlobalRng::rng);
    MvnConjMeanSampler(MvnModel *Mod,
                       const Vector &Mu0,
                       double Kappa,
                       RNG &seeding_rng = GlobalRng::rng);

    MvnConjMeanSampler *clone_to_new_host(Model *new_host) const override;
    double logpri() const override;  // p(mu|Sig)
    void draw() override;

   private:
    MvnModel *mvn;
    Ptr<VectorParams> mu0;
    Ptr<UnivParams> kappa;
  };
  //____________________________________________________________

  class MvnMeanSampler : public PosteriorSampler {
    // assumes y~N(mu, Sigma) with mu~N(mu0, Omega)
   public:
    MvnMeanSampler(MvnModel *Mod,
                   const Ptr<VectorParams> &Mu0,
                   const Ptr<SpdParams> &Omega,
                   RNG &seeding_rng = GlobalRng::rng);

    MvnMeanSampler(MvnModel *Mod,
                   const Ptr<MvnBase> &Pri,
                   RNG &seeding_rng = GlobalRng::rng);

    MvnMeanSampler(MvnModel *Mod,
                   const Vector &Mu0,
                   const SpdMatrix &Omega,
                   RNG &seeding_rng = GlobalRng::rng);
    MvnMeanSampler *clone_to_new_host(Model *new_host) const override;
    double logpri() const override;
    void draw() override;

   private:
    MvnModel *mvn;
    Ptr<MvnBase> mu_prior_;
  };
  //____________________________________________________________
}  // namespace BOOM
#endif  // BOOM_MVN_MEAN_SAMPLER_HPP

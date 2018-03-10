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

#ifndef BOOM_MLOGIT_RWM_SAMPLER_HPP
#define BOOM_MLOGIT_RWM_SAMPLER_HPP

#include "Models/Glm/MultinomialLogitModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class MlogitRwm : public PosteriorSampler {
   public:
    MlogitRwm(MultinomialLogitModel *mlm, const Ptr<MvnBase> &pri,
              RNG &seeding_rng = GlobalRng::rng);
    MlogitRwm(MultinomialLogitModel *mlm, const Vector &mu,
              const SpdMatrix &Ominv, RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

   private:
    MultinomialLogitModel *mlm_;
    Ptr<MvnBase> pri_;
    Vector mu, g, b, bstar;
    SpdMatrix H, ivar;
  };

}  // namespace BOOM

#endif  // BOOM_MLOGIT_RWM_SAMPLER_HPP

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
#ifndef BOOM_POISSON_GAMMA_METHOD_HPP
#define BOOM_POISSON_GAMMA_METHOD_HPP

#include "Models/GammaModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {
  class PoissonModel;

  class PoissonGammaSampler : public PosteriorSampler {
   public:
    PoissonGammaSampler(PoissonModel *p, const Ptr<GammaModel> &g,
                        RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;
    double alpha() const;
    double beta() const;
    void find_posterior_mode(double epsilon = 1e-5) override;
    bool can_find_posterior_mode() const override { return true; }

   private:
    PoissonModel *pois;
    Ptr<GammaModel> gam;
  };
}  // namespace BOOM
#endif  // BOOM_POISSON_GAMMA_METHOD_HPP

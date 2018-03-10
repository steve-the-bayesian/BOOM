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
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
namespace BOOM {

  class MultinomialProbitModel;
  class MvnModel;

  class MnpBetaSampler : public PosteriorSampler {
    // draws beta given Sigma for the multinomial probit model
    // assuming the prior beta|Sigma ~ N(b, Ominv)
   public:
    MnpBetaSampler(MultinomialProbitModel *Mod, const Ptr<MvnModel> &Pri,
                   RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;
    void fix_beta0(bool = true);

   private:
    MultinomialProbitModel *mnp;
    Ptr<MvnModel> pri;
    bool b0_fixed;
  };

}  // namespace BOOM

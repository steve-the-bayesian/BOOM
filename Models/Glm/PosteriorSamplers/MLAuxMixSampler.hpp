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

#ifndef BOOM_ML_AUX_MIX_SAMPLER_HPP
#define BOOM_ML_AUX_MIX_SAMPLER_HPP

#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "Models/Glm/PosteriorSamplers/MLVS.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class MultinomialLogitModel;
  class MvnBase;
  class ChoiceData;

  class MLAuxMixSampler : public PosteriorSampler {
    // draws the parameters of a multinomial logit model using the
    // approximate method from Fruhwirth-Schnatter and Fruhwirth, CSDA
    // 2007, 3508-3528.

    // this implementation only stores the complete data sufficient
    // statistics and some workspace.  It does not store the imputed
    // latent data.
   public:
    MLAuxMixSampler(MultinomialLogitModel *Mod, const Ptr<MvnBase> &Pri,
                    uint nthreads = 1, RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

   private:
    MultinomialLogitModel *mod_;
    Ptr<MvnBase> pri;
    Ptr<MLVS> sam;
  };

}  // namespace BOOM
#endif  // BOOM_ML_AUX_MIX_SAMPLER_HPP

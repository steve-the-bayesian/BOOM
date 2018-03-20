// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#include "Models/GammaModel.hpp"
#include "Models/GaussianModelBase.hpp"
#include "Models/PosteriorSamplers/GenericGaussianVarianceSampler.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

namespace BOOM {

  class SharedSigsqSampler : public PosteriorSampler {
    //  The model is y[i] | mu[s], sigsq ~ N(mu[s], sigsq), with mu[s] and
    //  sigq a priori independent (so the prior on mu is not informative
    //  for sigsq), and 1/sigsq ~ pri_

   public:
    SharedSigsqSampler(const std::vector<GaussianModelBase *> &models,
                       const Ptr<UnivParams> &Sigsq,
                       const Ptr<GammaModelBase> &pri,
                       RNG &seeding_rng = GlobalRng::rng);
    void draw() override;
    double logpri() const override;

   private:
    std::vector<GaussianModelBase *> models_;
    Ptr<UnivParams> sigsq_;
    Ptr<GammaModelBase> pri_;
    GenericGaussianVarianceSampler sigsq_sampler_;
  };

}  // namespace BOOM

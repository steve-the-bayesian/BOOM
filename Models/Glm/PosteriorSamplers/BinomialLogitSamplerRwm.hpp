// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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
#ifndef BOOM_BINOMIAL_LOGIT_SAMPLER_RWM_HPP_
#define BOOM_BINOMIAL_LOGIT_SAMPLER_RWM_HPP_

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/MvnBase.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

#include <functional>
#include "Samplers/MH_Proposals.hpp"
#include "Samplers/MetropolisHastings.hpp"

namespace BOOM {

  // A random walk Metropolis sampler for BinomialLogitModel's.  If nu
  // > 0 then the proposal is a multivariate T centered on the current
  // draw of beta.  If nu <= 0 the proposal is multivariate Gaussian.
  // This version assumes that all the elements of beta are included.
  // It does not work with spike and slab priors.
  class BinomialLogitSamplerRwm : public PosteriorSampler {
   public:
    BinomialLogitSamplerRwm(BinomialLogitModel *model,
                            const Ptr<MvnBase> &prior, double nu = 3,
                            RNG &seeding_rng = GlobalRng::rng);

    void draw() override;
    double logpri() const override;
    MvtRwmProposal *proposal() { return proposal_.get(); }
    const MvtRwmProposal *proposal() const { return proposal_.get(); }

    void set_chunk_size(int n);

   private:
    BinomialLogitModel *m_;
    Ptr<MvnBase> pri_;
    Ptr<MvtRwmProposal> proposal_;
    MetropolisHastings sam_;
  };

}  // namespace BOOM

#endif  //  BOOM_BINOMIAL_LOGIT_SAMPLER_RWM_HPP_

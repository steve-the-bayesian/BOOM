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

#include "Models/PosteriorSamplers/SharedSigsqSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  typedef SharedSigsqSampler SSS;

  SSS::SharedSigsqSampler(const std::vector<GaussianModelBase *> &models,
                          const Ptr<UnivParams> &sigsq,
                          const Ptr<GammaModelBase> &pri, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        models_(models),
        sigsq_(sigsq),
        pri_(pri),
        sigsq_sampler_(pri_) {}

  void SSS::draw() {
    double df = 0;
    double ss = 0;

    for (uint i = 0; i < models_.size(); ++i) {
      Ptr<GaussianSuf> suf = models_[i]->suf();
      double n = suf->n();
      df += n;
      double mu = models_[i]->mu();
      ss += suf->sumsq() + mu * suf->sum() + n * mu * mu;
    }

    double sigsq = sigsq_sampler_.draw(rng(), df, ss);
    sigsq_->set(sigsq);
  }

  double SSS::logpri() const {
    return sigsq_sampler_.log_prior(sigsq_->value());
  }

}  // namespace BOOM

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

#include "Models/PosteriorSamplers/CompositeSampler.hpp"
#include <cmath>
#include "distributions.hpp"

namespace BOOM {

  typedef CompositeSampler CS;
  typedef CompositeSamplerAdder CSA;
  typedef PosteriorSampler PS;

  CS::CompositeSampler(RNG &seeding_rng) : PosteriorSampler(seeding_rng) {}

  CS::CompositeSampler(const Ptr<PS> &p, double pr, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), samplers_(1, p), probs_(1, pr) {}

  CS::CompositeSampler(const std::vector<Ptr<PS> > &pv, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), samplers_(pv), probs_(pv.size(), 1.0) {}

  CS::CompositeSampler(const std::vector<Ptr<PS> > &pv, const Vector &pr,
                       RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), samplers_(pv), probs_(pr) {}

  CSA CS::add_sampler(const Ptr<PosteriorSampler> &s, double weight) {
    samplers_.push_back(s);
    probs_.push_back(weight);
    return CSA(this);
  }

  void CS::draw() { choose_sampler()->draw(); }

  double CS::logpri() const { return choose_sampler()->logpri(); }

  Ptr<PosteriorSampler> CS::choose_sampler() const {
    uint k = rmulti_mt(rng(), probs_);
    return samplers_[k];
  }
  //----------------------------------------------------------------------
  CSA::CompositeSamplerAdder(CS *cs_) : cs(cs_) {}

  CSA CSA::operator()(const Ptr<PosteriorSampler> &ps, double wgt) {
    cs->add_sampler(ps, wgt);
    return CSA(cs);
  }
}  // namespace BOOM

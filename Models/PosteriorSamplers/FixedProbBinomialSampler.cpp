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
#include "Models/PosteriorSamplers/FixedProbBinomialSampler.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  typedef FixedProbBinomialSampler FBS;
  FBS::FixedProbBinomialSampler(BinomialModel *mod,
                                double prob,
                                RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), m_(mod), p_(prob) {}

  FBS *FBS::clone_to_new_host(Model *new_host) const {
    return new FBS(dynamic_cast<BinomialModel *>(new_host),
                   p_,
                   rng());
  }

  void FBS::draw() { m_->set_prob(p_); }

  double FBS::logpri() const {
    double p = m_->prob();
    return p_ == p ? 0 : BOOM::negative_infinity();
  }
}  // namespace BOOM

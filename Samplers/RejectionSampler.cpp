// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Samplers/RejectionSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  RejectionSampler::RejectionSampler(const Target &log_target_density,
                                     const Ptr<DirectProposal> &proposal)
      : log_target_density_(log_target_density),
        proposal_(proposal),
        log_proposal_density_offset_(0.0),
        rejection_limit_(-1) {}

  Vector RejectionSampler::draw(RNG &rng) {
    std::int64_t rejection_count = 0;
    while (true) {
      if (rejection_limit_ > 0 && rejection_count++ > rejection_limit_) {
        return Vector(0);
      }
      Vector candidate = proposal_->draw(rng);
      double u = runif_mt(rng, 0, 1);
      while (u == 0) {
        // Avoid the fact that the RNG is on U[0, 1).
        u = runif_mt(rng, 0, 1);
      }
      if (log(u) < log_target_density_(candidate) - proposal_->logp(candidate) -
                       log_proposal_density_offset_) {
        return candidate;
      }
    }
  }

  void RejectionSampler::set_rejection_limit(std::int64_t limit) {
    rejection_limit_ = limit;
  }

  void RejectionSampler::set_log_proposal_density_offset(double offset) {
    log_proposal_density_offset_ = offset;
  }

}  // namespace BOOM

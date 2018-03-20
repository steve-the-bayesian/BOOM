/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Samplers/AdaptiveRandomWalkMetropolisSampler.hpp"
#include "distributions.hpp"

namespace BOOM {

  AdaptiveRandomWalkMetropolisSampler::AdaptiveRandomWalkMetropolisSampler(
      const LogDensity &log_density, double smoothing_weight_on_past, RNG *rng)
      : Sampler(rng),
        log_density_(log_density),
        smoothing_weight_(smoothing_weight_on_past),
        smoothed_sum_of_squares_(0),
        smoothed_sample_size_(0) {}

  Vector AdaptiveRandomWalkMetropolisSampler::draw(const Vector &old) {
    if (smoothed_sum_of_squares_.nrow() == 0) {
      smoothed_sum_of_squares_ = SpdMatrix(old.size());
      smoothed_sum_of_squares_.diag() = 1.0;
      smoothed_sample_size_ = 1.0;
    }

    SpdMatrix variance = smoothed_sum_of_squares_ / smoothed_sample_size_;
    Vector candidate = rmvn_mt(rng(), old, variance);
    double log_ratio = log_density_(candidate) - log_density_(old);
    double log_u = log(runif_mt(rng()));
    if (log_u < log_ratio) {
      update_proposal_distribution(candidate, old, true);
      return candidate;
    } else {
      update_proposal_distribution(candidate, old, false);
      return old;
    }
  }

  void AdaptiveRandomWalkMetropolisSampler::update_proposal_distribution(
      const Vector &cand, const Vector &old, bool accepted) {
    if (accepted) {
      Vector delta = cand - old;
      smoothed_sum_of_squares_ *= smoothing_weight_;
      smoothed_sum_of_squares_.add_outer(delta, 1 - smoothing_weight_);
      smoothed_sample_size_ *= smoothing_weight_;
      smoothing_weight_ += (1 - smoothing_weight_);
    } else {
      smoothed_sum_of_squares_ *= smoothing_weight_;
    }
  }

}  // namespace BOOM

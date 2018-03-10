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

#include "Samplers/ScalarLangevinSampler.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  ScalarLangevinSampler::ScalarLangevinSampler(
      const Ptr<dScalarTargetFun> &logf, double initial_step_size, RNG *rng)
      : ScalarSampler(rng), logf_(logf), adapt_(false) {
    set_step_size(initial_step_size);
  }

  double ScalarLangevinSampler::draw(double current_x) {
    if (adapt_) {
      if (consecutive_rejects_ > 10) {
        // step size needs to get smaller
        set_step_size(step_size_ * runif_mt(rng(), .9, 1.0));
      } else if (consecutive_accepts_ > 10) {
        set_step_size(step_size_ * runif_mt(rng(), 1.0, 1.1));
      }
    }

    double current_gradient = 0;
    double logp_current = logf(current_x, current_gradient);
    double proposal_mean = current_x + 0.5 * current_gradient * step_size_;
    double proposal = rnorm_mt(rng(), proposal_mean, sd_);

    double proposal_gradient = 0;
    double logp_proposal = logf(proposal, proposal_gradient);
    double reverse_mean = proposal + 0.5 * proposal_gradient * step_size_;

    double log_acceptance_ratio =
        logp_proposal - dnorm(proposal, proposal_mean, sd_, true) -
        logp_current + dnorm(current_x, reverse_mean, sd_, true);
    if (log(runif_mt(rng())) < log_acceptance_ratio) {
      consecutive_rejects_ = 0;
      ++consecutive_accepts_;
      return proposal;
    } else {
      consecutive_accepts_ = 0;
      ++consecutive_rejects_;
      return current_x;
    }
  }

  void ScalarLangevinSampler::set_step_size(double step_size) {
    if (step_size <= 0) {
      report_error("step_size must be positive");
    }
    step_size_ = step_size;
    sd_ = sqrt(step_size);
  }

  void ScalarLangevinSampler::allow_adaptation(bool okay_to_adapt) {
    adapt_ = okay_to_adapt;
  }

}  // namespace BOOM

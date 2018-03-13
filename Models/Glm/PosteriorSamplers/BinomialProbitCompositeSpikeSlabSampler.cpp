// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/BinomialProbitCompositeSpikeSlabSampler.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    typedef BinomialProbitCompositeSpikeSlabSampler BPCSSS;
  }  // namespace

  BPCSSS::BinomialProbitCompositeSpikeSlabSampler(
      BinomialProbitModel *model, const Ptr<MvnBase> &slab,
      const Ptr<VariableSelectionPrior> &spike, int clt_threshold,
      double proposal_df, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        slab_(slab),
        spike_(spike),
        spike_slab_sampler_(model_, slab_, spike_, clt_threshold, seeding_rng),
        tim_(model_, slab_, proposal_df, seeding_rng),
        sampling_weights_{.5, .5} {}

  double BPCSSS::logpri() const { return spike_slab_sampler_.logpri(); }

  void BPCSSS::draw() {
    try {
      uint which_sampler = rmulti_mt(rng(), sampling_weights_);
      if (which_sampler == 0) {
        try {
          spike_slab_sampler_.draw();
        } catch (...) {
          tim_.draw();
        }
      } else if (which_sampler == 1) {
        try {
          tim_.draw();
        } catch (...) {
          spike_slab_sampler_.draw();
        }
      }
    } catch (...) {
    }
  }

  void BPCSSS::set_sampling_weights(const Vector &weights) {
    if (weights.size() != 2) {
      report_error("Sampling weight vector must have 2 elements.");
    }
    if (weights.min() < 0) {
      report_error("Negative weights not allowed.");
    }
    double total = sum(weights);
    if (!std::isfinite(total)) {
      report_error("Infinite or NaN values in weights.");
    }
    sampling_weights_ = weights / total;
  }

}  // namespace BOOM

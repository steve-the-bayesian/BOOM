/*
  Copyright (C) 2005-2021 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/StateSpace/StateModels/PosteriorSamplers/GeneralSeasonalLLTPosteriorSampler.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {
    using GSLLT = GeneralSeasonalLLT;
    using GSLLTPS = GeneralSeasonalLLTPosteriorSampler;
  }  // namespace

  GSLLTPS::GeneralSeasonalLLTPosteriorSampler(
      GSLLT *model,
      const std::vector<Ptr<WishartModel>> &priors,
      RNG &seeding_rng)
      : PosteriorSampler(seeding_rng),
        model_(model),
        priors_(priors)
  {
    if (model_->nseasons() != priors_.size()) {
      report_error("There should be one Wishart prior for each season.");
    }
    for (int i = 0; i < model_->nseasons(); ++i) {
      subordinate_samplers_.push_back(
          new ZeroMeanMvnConjSampler(
              model_->subordinate_model(i),
              priors_[i],
              rng()));
    }
  }

  void GSLLTPS::draw() {
    for (auto &sam : subordinate_samplers_) {
      sam->draw();
    }
  }

  double GSLLTPS::logpri() const {
    double ans = 0;
    for (const auto &sam : subordinate_samplers_) {
      ans += sam->logpri();
    }
    return ans;
  }


}  // namespace BOOM

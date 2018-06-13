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

#include "Models/StateSpace/PosteriorSamplers/AggregatedStateSpacePosteriorSampler.hpp"

namespace BOOM {
  namespace {
    using ASSPS = AggregatedStateSpacePosteriorSampler;
  }

  ASSPS::AggregatedStateSpacePosteriorSampler(
      AggregatedStateSpaceRegression *model, RNG &seeding_rng)
      : PosteriorSampler(seeding_rng), m_(model) {}

  void ASSPS::draw() {
    m_->impute_state(rng());
    m_->regression_model()->sample_posterior();

    // Don't re-sample the regression model (in position 0).
    for (int s = 1; s < m_->number_of_state_models(); ++s) {
      m_->state_model(s)->sample_posterior();
    }
  }

  double ASSPS::logpri() const {
    double ans = m_->regression_model()->logpri();
    for (int s = 1; s < m_->number_of_state_models(); ++s) {
      ans += m_->state_model(s)->logpri();
    }
    return ans;
  }

}  // namespace BOOM

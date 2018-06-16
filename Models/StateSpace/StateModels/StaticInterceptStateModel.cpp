// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "Models/StateSpace/StateModels/StaticInterceptStateModel.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "distributions.hpp"

namespace BOOM {
  namespace {
    using SISM = StaticInterceptStateModel;
  }
  
  StaticInterceptStateModel::StaticInterceptStateModel()
      : state_transition_matrix_(new IdentityMatrix(1)),
        state_variance_matrix_(new ZeroMatrix(1)),
        observation_matrix_(1),
        initial_state_mean_(1, 0.0),
        initial_state_variance_(1, 1.0) {
    observation_matrix_[0] = 1.0;
  }

  void StaticInterceptStateModel::simulate_initial_state(RNG &rng,
                                                         VectorView eta) const {
    eta[0] = rnorm_mt(rng, initial_state_mean_[0],
                      sqrt(initial_state_variance_(0, 0)));
  }

  void StaticInterceptStateModel::set_initial_state_mean(double mean) {
    initial_state_mean_[0] = mean;
  }

  void StaticInterceptStateModel::set_initial_state_variance(double variance) {
    if (variance < 0) {
      report_error("Initial state variance must be non-negative.");
    }
    initial_state_variance_(0, 0) = variance;
  }

}  // namespace BOOM

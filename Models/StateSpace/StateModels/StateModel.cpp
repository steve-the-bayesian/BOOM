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

#include "Models/StateSpace/StateModels/StateModel.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  StateModelBase::StateModelBase()
      : index_(-1)
  {}
  
  void StateModelBase::update_complete_data_sufficient_statistics(
      int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error(
        "update_complete_data_sufficient_statistics does not work "
        "for this StateModel subclass.");
  }

  void StateModelBase::increment_expected_gradient(
      VectorView gradient, int t, const ConstVectorView &state_error_mean,
      const ConstSubMatrix &state_error_variance) {
    report_error(
        "increment_expected_gradient does not work for "
        "this StateModel subclass.");
  }

  void StateModelBase::simulate_initial_state(RNG &rng, VectorView eta) const {
    if (eta.size() != state_dimension()) {
      std::ostringstream err;
      err << "output vector 'eta' has length " << eta.size()
          << " in StateModel::simulate_initial_state.  Expected length "
          << state_dimension();
      report_error(err.str());
    }
    eta = rmvn_mt(rng, initial_state_mean(), initial_state_variance());
  }

  void StateModelBase::observe_initial_state(const ConstVectorView &state) {}

  
}  // namespace BOOM

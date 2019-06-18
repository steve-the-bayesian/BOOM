/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "gtest/gtest.h"
#include "test_utils/test_utils.hpp"
#include "Models/StateSpace/StateModels/test_utils/StaticInterceptTestModule.hpp"
#include "distributions.hpp"
#include "stats/AsciiDistributionCompare.hpp"

namespace BOOM {
  namespace StateSpaceTesting {
    StaticInterceptTestModule::StaticInterceptTestModule(double intercept)
        : intercept_(intercept),
          intercept_model_(new StaticInterceptStateModel)
    {
      intercept_model_->set_initial_state_mean(intercept_);
      intercept_model_->set_initial_state_variance(square(100.0));
    }

    void StaticInterceptTestModule::SimulateData(int time_dimension) {
      state_.assign(time_dimension, intercept_);
    }

    void StaticInterceptTestModule::CreateObservationSpace(int niter) {
      intercept_draws_.resize(niter);
    }

    void StaticInterceptTestModule::ObserveDraws(
        const ScalarStateSpaceModelBase &model) {
      auto state = CurrentState(model);
      intercept_draws_[cursor()] = state(0, 0);
    }

    void StaticInterceptTestModule::Check() {
      EXPECT_GT(var(intercept_draws_), 0.0)
          << "Static intercept is not being updated.";
      EXPECT_TRUE(CheckMcmcVector(intercept_draws_, intercept_))
          << AsciiDistributionCompare(intercept_draws_, intercept_);
    }
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM

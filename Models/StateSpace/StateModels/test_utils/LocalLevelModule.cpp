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

#include "Models/StateSpace/StateModels/test_utils/LocalLevelModule.hpp"

#include "gtest/gtest.h"

#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "test_utils/test_utils.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    LocalLevelModule::LocalLevelModule(double level_sd, double initial_level)
        : level_sd_(level_sd),
          initial_level_(initial_level),
          trend_model_(new LocalLevelStateModel),
          // adapter_(new DynamicInterceptStateModelAdapter(trend_model_)),
          level_precision_prior_(new ChisqModel(1.0, level_sd_)),
          level_precision_sampler_(new ZeroMeanGaussianConjSampler(
              trend_model_.get(), level_precision_prior_))
    {
      trend_model_->set_method(level_precision_sampler_);
      trend_model_->set_initial_state_mean(initial_level);
      trend_model_->set_initial_state_variance(level_sd_);
    }

    void LocalLevelModule::SimulateData(int time_dimension) {
      trend_.resize(time_dimension);
      double level = initial_level_;
      for (int i = 0; i < time_dimension; ++i) {
        trend_[i] = level;
        level += rnorm(0, level_sd_);
      }
    }

    void LocalLevelModule::CreateObservationSpace(int niter) {
      trend_draws_.resize(niter, trend_.size());
      sigma_level_draws_.resize(niter);
    }

    void LocalLevelModule::ObserveDraws(
        const ScalarStateSpaceModelBase &model) {
      auto state = CurrentState(model);
      trend_draws_.row(cursor()) = state.row(0);
      sigma_level_draws_[cursor()] = trend_model_->sigma();
    }

    void LocalLevelModule::Check() {
      auto status = CheckMcmcMatrix(
          trend_draws_, trend_, .95, true, "trend.txt");
      EXPECT_TRUE(status.ok)
          <<  "Local level failed to cover." << std::endl
          << status;

      EXPECT_GT(var(sigma_level_draws_), 0)
          << "sigma level draws had zero variance";
      EXPECT_TRUE(CheckMcmcVector(sigma_level_draws_, level_sd_, .95,
                                  "sigma-level.txt"))
          << "Innovation SD for local level model "
          << "did not cover true value." << std::endl
          << AsciiDistributionCompare(sigma_level_draws_, level_sd_);
    }
    
  }  // namespace StateSpaceTesting
}  //namespace BOOM


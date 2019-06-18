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

#include "Models/StateSpace/StateModels/test_utils/SeasonalTestModule.hpp"
#include "LinAlg/Matrix.hpp"
#include "distributions.hpp"
#include "stats/AsciiDistributionCompare.hpp"


namespace BOOM {
  namespace StateSpaceTesting {

    SeasonalTestModule::SeasonalTestModule(double sd,
                                           const Vector &pattern,
                                           int season_duration)
        : sd_(sd),
          initial_pattern_(pattern),
          season_duration_(season_duration),
          seasonal_model_(new SeasonalStateModel(pattern.size(),
                                                 season_duration_)),
          // adapter_(new DynamicInterceptStateModelAdapter(seasonal_model_)),
          precision_prior_(new ChisqModel(1.0, sd_)),
          sampler_(new ZeroMeanGaussianConjSampler(
              seasonal_model_.get(), precision_prior_))
    {
      seasonal_model_->set_method(sampler_);
      state_dim_ = initial_pattern_.size() - 1;
      seasonal_model_->set_initial_state_mean(
          pattern_to_state(initial_pattern_));
      seasonal_model_->set_initial_state_variance(
          SpdMatrix(state_dim_, sd_ * sd_));
    }
    
    SeasonalTestModule::SeasonalTestModule(double sd,
                                           int nseasons,
                                           int season_duration)
        : SeasonalTestModule(sd, random_initial_pattern(nseasons),
                             season_duration) {}
    
    void SeasonalTestModule::SimulateData(int time_dimension) {
      Vector state = pattern_to_state(initial_pattern_);
      Matrix transition(state.size(), state.size(), 0.0);
      transition.row(0) = -1;
      transition.subdiag(1) = 1.0;
      seasonal_.resize(time_dimension);
      for (int t = 0; t < time_dimension; ++t) {
        seasonal_[t] = state[0];
        for (int j = 1; j < season_duration_; ++j) {
          ++t;
          if (t >= time_dimension) break;
          seasonal_[t] = state[0];
        }
        state = transition * state;
        state[0] += rnorm_mt(GlobalRng::rng, 0, sd_);
      }
    }

    void SeasonalTestModule::CreateObservationSpace(int niter) {
      seasonal_draws_.resize(niter, seasonal_.size());
      sigma_draws_.resize(niter);
    }

    void SeasonalTestModule::ObserveDraws(
        const ScalarStateSpaceModelBase &model) {
      auto state = CurrentState(model);
      seasonal_draws_.row(cursor()) = state.row(0);
      sigma_draws_[cursor()] = seasonal_model_->sigma();
    }

    void SeasonalTestModule::Check() {
      auto status = CheckMcmcMatrix(
          seasonal_draws_, seasonal_, .95, true, "seasonal.txt");
      EXPECT_TRUE(status.ok)
          <<  "Seasonal state component failed to cover." << std::endl
          << status;

      EXPECT_GT(var(sigma_draws_), 0)
          << "sigma level draws had zero variance";
      EXPECT_TRUE(CheckMcmcVector(sigma_draws_, sd_, .99,
                                  "sigma-seasonal.txt"))
          << "Innovation SD for seasonal component "
          << "did not cover true value." << std::endl
          << AsciiDistributionCompare(sigma_draws_, sd_);
    }
    
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM

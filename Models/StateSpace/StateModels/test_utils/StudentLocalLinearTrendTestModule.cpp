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
#include "Models/StateSpace/StateModels/test_utils/StudentLocalLinearTrendTestModule.hpp"
#include "distributions.hpp"
#include "stats/AsciiDistributionCompare.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    StudentLocalLinearTrendTestModule::StudentLocalLinearTrendTestModule(
        double level_sd, double initial_level, double nu_level,
        double slope_sd, double initial_slope, double nu_slope)
        : level_sd_(level_sd),
          initial_level_(initial_level),
          nu_level_(nu_level),
          slope_sd_(slope_sd),
          initial_slope_(initial_slope),
          nu_slope_(nu_slope),
          trend_model_(new StudentLocalLinearTrendStateModel(
              level_sd_, nu_level_, slope_sd_, nu_slope_)),
          // adapter_(new DynamicInterceptStateModelAdapter(trend_model_)),
          level_precision_prior_(new ChisqModel(2.0, level_sd_)),
          nu_level_prior_(new UniformModel(1.0, 100.0)),
          slope_precision_prior_(new ChisqModel(2.0, slope_sd_)),
          nu_slope_prior_(new UniformModel(1.0, 100.0)),
          trend_sampler_(new StudentLocalLinearTrendPosteriorSampler(
              trend_model_.get(),
              level_precision_prior_,
              nu_level_prior_,
              slope_precision_prior_,
              nu_slope_prior_)) {
      trend_model_->set_method(trend_sampler_);
      trend_model_->set_initial_state_mean(Vector{initial_level_, initial_slope_});
      SpdMatrix initial_variance(2);
      initial_variance.diag() = square(Vector{level_sd_, slope_sd_});
      trend_model_->set_initial_state_variance(initial_variance);
    }

    void StudentLocalLinearTrendTestModule::SimulateData(int time_dimension) {
      trend_.resize(time_dimension);
      double level = initial_level_;
      double slope = initial_slope_;
      for (int t = 0; t < time_dimension; ++t) {
        trend_[t] = level;
        level += slope + rstudent_mt(GlobalRng::rng, 0, level_sd_, nu_level_);
        slope += rstudent_mt(GlobalRng::rng, 0, slope_sd_, nu_slope_);
      }
    }

    void StudentLocalLinearTrendTestModule::CreateObservationSpace(int niter) {
      trend_draws_.resize(niter, trend_.size());
      sigma_level_draws_.resize(niter);
      sigma_slope_draws_.resize(niter);
      nu_level_draws_.resize(niter);
      nu_slope_draws_.resize(niter);
    }

    void StudentLocalLinearTrendTestModule::ObserveDraws(
        const ScalarStateSpaceModelBase &model) {
      auto state = CurrentState(model);
      trend_draws_.row(cursor()) = state.row(0);
      sigma_level_draws_[cursor()] = trend_model_->sigma_level();
      sigma_slope_draws_[cursor()] = trend_model_->sigma_slope();
      nu_level_draws_[cursor()] = trend_model_->nu_level();
      nu_slope_draws_[cursor()] = trend_model_->nu_slope();
    }

    void StudentLocalLinearTrendTestModule::Check() {
      EXPECT_EQ("", CheckStochasticProcess(
          trend_draws_, trend_, .95, .1, 0.5, "trend.txt"))
          <<  "Student local linear trend failed to cover.";

      EXPECT_GT(var(sigma_level_draws_), 0)
          << "sigma level draws had zero variance";
      EXPECT_TRUE(CheckMcmcVector(sigma_level_draws_, level_sd_, .95,
                                  "sigma-level.txt"))
          << "Innovation SD for student local linear trend model, level "
          << "component did not cover true value of " << level_sd_
          << "." << std::endl
          << AsciiDistributionCompare(sigma_level_draws_, level_sd_);

      EXPECT_GT(var(nu_level_draws_), 0)
          << "nu parameter for level component is not being updated";
      EXPECT_TRUE(CheckMcmcVector(nu_level_draws_, nu_level_, .95,
                                  "nu-level.txt"))
          << "Nu parameter for local linear trend model, level component "
          << "did not cover the true value of " << nu_level_ <<"." << std::endl
          << AsciiDistributionCompare(nu_level_draws_, nu_level_);


      EXPECT_GT(var(sigma_slope_draws_), 0)
          << "sigma slop draws had zero variance";
      EXPECT_TRUE(CheckMcmcVector(sigma_slope_draws_, slope_sd_, .95,
                                  "sigma-slope.txt"))
          << "Innovation SD for the slope portion of the local linear trend "
          << "model did not cover true value of " << slope_sd_
          << "." << std::endl
          << AsciiDistributionCompare(sigma_slope_draws_, slope_sd_);

      EXPECT_GT(var(nu_slope_draws_), 0)
          << "nu parameter for slope component is not being updated";
      EXPECT_TRUE(CheckMcmcVector(nu_slope_draws_, nu_slope_, .95,
                                  "nu-slope.txt"))
          << "Nu parameter for local linear trend model, slope component "
          << "did not cover true value of " << nu_slope_
          << "." << std::endl
          << AsciiDistributionCompare(nu_slope_draws_, nu_slope_);
    }

  }  // namespace StateSpaceTesting
}  // namespace BOOM

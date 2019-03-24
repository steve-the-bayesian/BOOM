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
#include "Models/StateSpace/StateModels/test_utils/SemilocalLinearTrendTestModule.hpp"
#include "distributions.hpp"
#include "stats/AsciiDistributionCompare.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    namespace {
      using Module = SemilocalLinearTrendTestModule;
    }  // namespace

    Module::SemilocalLinearTrendTestModule(
        double level_sd, double initial_level,
        double slope_sd, double initial_slope,
        double slope_mean, double slope_ar)
        : level_sd_(level_sd),
          initial_level_(initial_level),
          slope_sd_(slope_sd),
          initial_slope_(initial_slope),
          slope_mean_(slope_mean),
          slope_ar_(slope_ar),
          level_model_(new ZeroMeanGaussianModel(level_sd_)),
          level_precision_prior_(new ChisqModel(1.0, level_sd_)),
          level_sampler_(new ZeroMeanGaussianConjSampler(
              level_model_.get(), level_precision_prior_)),
          slope_model_(new NonzeroMeanAr1Model(
              slope_mean_, slope_ar_, slope_sd_)),
          slope_mean_prior_(new GaussianModel(
              slope_mean, square(10.0))),
          slope_ar_prior_(new GaussianModel(0.0, square(.5))),
          slope_precision_prior_(new ChisqModel(1.0, slope_sd_)),
          slope_sampler_(new NonzeroMeanAr1Sampler(
              slope_model_.get(), slope_mean_prior_,
              slope_ar_prior_, slope_precision_prior_)),
          trend_model_(new SemilocalLinearTrendStateModel(
              level_model_, slope_model_))
          // adapter_(new DynamicInterceptStateModelAdapter(trend_model_))
    {
      level_model_->set_method(level_sampler_);
      slope_model_->set_method(slope_sampler_);
      trend_model_->set_initial_level_mean(initial_level_);
      trend_model_->set_initial_slope_mean(initial_slope_);
      trend_model_->set_initial_level_sd(level_sd_);
      trend_model_->set_initial_slope_sd(slope_sd_);
    }

    void Module::SimulateData(int time_dimension) {
      trend_.resize(time_dimension);
      double level = initial_level_;
      double slope = initial_slope_;

      for (int t = 0; t < time_dimension; ++t) {
        trend_[t] = level;
        level += slope + rnorm_mt(GlobalRng::rng, 0, level_sd_);
        slope = slope_mean_ + slope_ar_ * (slope - slope_mean_)
            + rnorm_mt(GlobalRng::rng, 0, slope_sd_);
      }
    }

    void Module::CreateObservationSpace(int niter) {
      trend_draws_.resize(niter, trend_.size());
      sigma_level_draws_.resize(niter);
      sigma_slope_draws_.resize(niter);
      slope_mean_draws_.resize(niter);
      slope_ar_draws_.resize(niter);
    }

    void Module::ObserveDraws(const ScalarStateSpaceModelBase &model) {
      auto state = CurrentState(model);
      trend_draws_.row(cursor()) = state.row(0);
      sigma_level_draws_[cursor()] = level_model_->sigma();
      sigma_slope_draws_[cursor()] = slope_model_->sigma();
      slope_mean_draws_[cursor()] = slope_model_->mu();
      slope_ar_draws_[cursor()] = slope_model_->phi();
    }
    
    void Module::Check() {
      auto status = CheckMcmcMatrix(trend_draws_, trend_);
      EXPECT_TRUE(status.ok) << "Semilocal linear trend component did not cover "
                             << "true trend." << std::endl
                             << status;

      EXPECT_GT(var(sigma_level_draws_), 0.0)
          << "Sigma level is constant.";
      EXPECT_TRUE(CheckMcmcVector(sigma_level_draws_, level_sd_))
          << "Sigma level did not cover" << std::endl
          << AsciiDistributionCompare(sigma_level_draws_, level_sd_);

      EXPECT_GT(var(sigma_slope_draws_), 0.0)
          << "Sigma slope is not being drawn.";
      EXPECT_TRUE(CheckMcmcVector(sigma_slope_draws_, slope_sd_))
          << "Sigma slope did not cover" << std::endl
          << AsciiDistributionCompare(sigma_slope_draws_, slope_sd_);

      EXPECT_GT(var(slope_mean_draws_), 0.0)
          << "Slope mean is not being drawn.";
      EXPECT_TRUE(CheckMcmcVector(slope_mean_draws_, slope_mean_))
          << "Slope long term mean did not cover" << std::endl
          << AsciiDistributionCompare(slope_mean_draws_, slope_mean_);

      EXPECT_GT(var(slope_ar_draws_), 0.0)
          << "Slope AR is not being drawn.";
      EXPECT_TRUE(CheckMcmcVector(slope_ar_draws_, slope_ar_))
          << "Slope AR coefficient did not cover" << std::endl
          << AsciiDistributionCompare(slope_ar_draws_, slope_ar_);
    }
    
  } // namespace StateSpaceTesting
}  // namespace BOOM

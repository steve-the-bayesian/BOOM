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

#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "gtest/gtest.h"

#include "Models/StateSpace/PosteriorSamplers/DynamicInterceptRegressionPosteriorSampler.hpp"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "cpputil/report_error.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    DynamicInterceptTestFramework::DynamicInterceptTestFramework(
        const Vector &coefficients,
        double observation_sd,
        double poisson_observation_rate)
        : true_beta_(coefficients),
          observation_sd_(observation_sd),
          poisson_rate_(poisson_observation_rate),
          model_(new DynamicInterceptRegressionModel(true_beta_.size())),
          residual_precision_prior_(new ChisqModel(1.0, observation_sd_)),
          xtx_(true_beta_.size(), 0.0),
          total_nobs_(0.0)
    {}

    void DynamicInterceptTestFramework::SimulateData(int time_dimension) {
      state_modules().SimulateData(time_dimension);
      Vector state = state_modules().StateContribution();
      data_.clear();
      for (int t = 0; t < time_dimension; ++t) {
        int nobs = 1 + rpois(poisson_rate_);
        Matrix predictors(nobs, true_beta_.size());
        predictors.randomize();
        Vector response = predictors * true_beta_;
        response += state[t];
        for (int j = 0; j < nobs; ++j) {
          response[j] += rnorm_mt(GlobalRng::rng, 0, observation_sd_);
        }
        NEW(StateSpace::TimeSeriesRegressionData, data_point)(
            response, predictors, Selector(response.size(), true));
        data_.push_back(data_point);
        xtx_ += predictors.inner();
        total_nobs_ += nobs;
      }
    }

    void DynamicInterceptTestFramework::BuildModel() {
      state_modules().ImbueState(*model_);
      for (const auto &data_point : data_) {
        model_->add_data(data_point);
      }
      NEW(MvnGivenScalarSigma, coefficient_prior)(
          .01 * xtx_ / total_nobs_, model_->observation_model()->Sigsq_prm());
      NEW(RegressionConjSampler, regression_sampler)(
          model_->observation_model(),
          coefficient_prior,
          residual_precision_prior_);
      model_->observation_model()->set_method(regression_sampler);
      NEW(DynamicInterceptRegressionPosteriorSampler, sampler)(model_.get());
      model_->set_method(sampler);
    }

    void DynamicInterceptTestFramework::CreateObservationSpace(int niter) {
      state_modules().CreateObservationSpace(niter);
      coefficient_draws_.resize(niter, true_beta_.size());
      sigma_obs_draws_.resize(niter);
    }

    void DynamicInterceptTestFramework::RunMcmc(int niter) {
      for (int i = 0; i < niter; ++i) {
        model_->sample_posterior();
        sigma_obs_draws_[i] = model_->observation_model()->sigma();
        coefficient_draws_.row(i) = model_->observation_model()->Beta();
        state_modules().ObserveDraws(*model_);
      }
    }

    void DynamicInterceptTestFramework::Check() {
      state_modules().Check();
      EXPECT_GT(var(sigma_obs_draws_), 0.0)
          << "sigma_obs_draws has zero variance.";
      EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws_, observation_sd_,
                                  .95, "sigma-obs.txt"))
          << AsciiDistributionCompare(sigma_obs_draws_, observation_sd_);
              
      auto status = CheckMcmcMatrix(coefficient_draws_, true_beta_,
                                    .95, true, "beta.txt");
      EXPECT_TRUE(status.ok) << "Coefficients did not cover" << status;
    }
  }
}  // namespace BOOM

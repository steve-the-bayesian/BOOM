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

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    StateSpaceTestFramework::StateSpaceTestFramework(double observation_sd)
        : observation_sd_(observation_sd),
          model_(new StateSpaceModel),
          residual_precision_prior_(new ChisqModel(1.0, observation_sd_)),
          residual_precision_sampler_(new ZeroMeanGaussianConjSampler(
              model_->observation_model(), residual_precision_prior_))
    {
      model_->observation_model()->set_method(residual_precision_sampler_);
      NEW(StateSpacePosteriorSampler, sampler)(model_.get());
      model_->set_method(sampler);
    }

    void StateSpaceTestFramework::SimulateData(int time_dimension) {
      if (state_modules().empty()) {
        report_error("Add state modules before calling SimulateData.");
      }
      state_modules().SimulateData(time_dimension);
      data_ = state_modules().StateContribution();
      for (int t = 0; t < time_dimension; ++t) {
        data_[t] += rnorm_mt(GlobalRng::rng, 0, observation_sd_);
      }
    }

    void StateSpaceTestFramework::BuildModel() {
      if (state_modules().empty()) {
        report_error("Add state modules before calling BuildModel.");
      }
      state_modules().ImbueState(*model_);
      for (int i = 0; i < data_.size(); ++i) {
        NEW(StateSpace::MultiplexedDoubleData, data_point)();
        data_point->add_data(new DoubleData(data_[i]));
        model_->add_data(data_point);
      }
    }

    void StateSpaceTestFramework::CreateObservationSpace(int niter) {
      state_modules().CreateObservationSpace(niter);
      sigma_obs_draws_.resize(niter);
    }

    void StateSpaceTestFramework::RunMcmc(int niter) {
      for (int i = 0; i < niter; ++i) {
        model_->sample_posterior();
        sigma_obs_draws_[i] = model_->observation_model()->sigma();
        state_modules().ObserveDraws(*model_);
      }
    }

    void StateSpaceTestFramework::Check() {
      state_modules().Check();
      EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws_, observation_sd_,
                                  .95, "StateSpaceModel-sigma-obs.txt"))
          << AsciiDistributionCompare(sigma_obs_draws_, observation_sd_);
    }
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM

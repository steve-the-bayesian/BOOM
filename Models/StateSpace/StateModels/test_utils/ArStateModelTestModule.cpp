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

#include "Models/StateSpace/StateModels/test_utils/ArStateModelTestModule.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    ArStateModelTestModule::ArStateModelTestModule(
        const Vector &ar_coefficients, double sd)
        : ar_coefficients_(ar_coefficients),
          sd_(sd),
          trend_model_(new ArStateModel(ar_coefficients.size()))
    {
        NEW(ChisqModel, precision_prior)(1.0, sd);
        NEW(ArPosteriorSampler, sampler)(trend_model_.get(), precision_prior);
        trend_model_->set_method(sampler);
        if (!ArModel::check_stationary(ar_coefficients)) {
          report_error("AR coefficients give a non-stationary model.");
        }
        trend_model_->set_phi(ar_coefficients);
        trend_model_->set_sigma(sd);
        trend_model_->set_initial_state_mean(
            Vector(trend_model_->state_dimension(), 0.0));
        SpdMatrix initial_variance(trend_model_->state_dimension(), 0.0);
        initial_variance.set_diag(trend_model_->stationary_variance());
        trend_model_->set_initial_state_variance(initial_variance);
    }

    void ArStateModelTestModule::SimulateData(int time_dimension) {
      trend_ = trend_model_->ArModel::simulate(time_dimension);
    }

    void ArStateModelTestModule::CreateObservationSpace(int niter) {
      trend_draws_.resize(niter, trend_.size());
      sigma_draws_.resize(niter);
      coefficient_draws_.resize(niter, ar_coefficients_.size());
    }

    void ArStateModelTestModule::Check() {
      auto status = CheckMcmcMatrix(trend_draws_, trend_);
      EXPECT_TRUE(status.ok)
          << "ArStateModel trend did not cover true value."
          << std::endl << status;

      EXPECT_TRUE(CheckMcmcVector(sigma_draws_, sd_))
          << "AR residual SD did not cover " << endl
          << AsciiDistributionCompare(sigma_draws_, sd_);

      status = CheckMcmcMatrix(coefficient_draws_, ar_coefficients_);
      EXPECT_TRUE(status.ok) << "AR coefficients did not cover." << std::endl
                             << status;
    }

  }  // namespace StateSpaceTesting
}  // namespace BOOM

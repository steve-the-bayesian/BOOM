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

#include "Models/StateSpace/StateModels/test_utils/TrigTestModule.hpp"
#include "cpputil/Constants.hpp"
#include "distributions.hpp"
#include "stats/AsciiDistributionCompare.hpp"

namespace BOOM {
  namespace StateSpaceTesting {
    TrigTestModule::TrigTestModule(double period, const Vector &frequencies, double sd)
        : period_(period),
          frequencies_(frequencies),
          sd_(sd),
          trig_model_(new TrigStateModel(period_, frequencies_)),
          // adapter_(new DynamicInterceptStateModelAdapter(trig_model_)),
          precision_prior_(new ChisqModel(1.0, sd_)),
          sampler_(new ZeroMeanGaussianConjSampler(
              trig_model_->error_distribution(),
              precision_prior_))
    {
      trig_model_->set_method(sampler_);
      int state_dim = trig_model_->state_dimension();
      trig_model_->set_initial_state_mean(Vector(state_dim, 0.0));
      SpdMatrix initial_variance(state_dim, 0.0);
      initial_variance.diag() = square(sd_);
      trig_model_->set_initial_state_variance(initial_variance);
    }

    void TrigTestModule::SimulateData(int time_dimension) {
      trig_.resize(time_dimension);
      std::vector<Matrix> rotations;
      std::vector<Vector> state;
      for (double freq : frequencies_) {
        Matrix rotation(2, 2);
        double cosine = cos(2 * Constants::pi * freq / period_);
        double sine = sin(2 * Constants::pi * freq / period_);
        rotation.diag() = cosine;
        rotation(0, 1) = sine;
        rotation(1, 0) = -sine;
        rotations.push_back(rotation);

        Vector state_contribution(2);
        state_contribution.randomize();
        state.push_back(state_contribution);        
      }

      for (int t = 0; t < time_dimension; ++t) {
        trig_[t] = 0;
        for (int j = 0; j < state.size(); ++j) {
          trig_[t] += state[j][0];
          state[j] = rotations[j] * state[j];
          state[j][0] += rnorm_mt(GlobalRng::rng, 0, sd_);
          state[j][1] += rnorm_mt(GlobalRng::rng, 0, sd_);
        }
      }
    }

    void TrigTestModule::CreateObservationSpace(int niter) {
      trig_draws_.resize(niter, trig_.size());
      sigma_draws_.resize(niter);
    }

    void TrigTestModule::ObserveDraws(
        const ScalarStateSpaceModelBase &model) {
      auto state = CurrentState(model);
      trig_draws_.row(cursor()) = 0;
      for (int i = 0; i < state.nrow(); i += 2) {
        trig_draws_.row(cursor()) += state.row(i);
      }
      sigma_draws_[cursor()] = trig_model_->error_distribution()->sigma();
    }

    void TrigTestModule::Check() {
      auto status = CheckMcmcMatrix(trig_draws_, trig_);
      EXPECT_TRUE(status.ok) << "Trig state model failed to cover state."
                             << std::endl << status;

      EXPECT_TRUE(CheckMcmcVector(sigma_draws_, sd_))
          << "Innovation SD for trig state model failed to cover true value of "
          << sd_ << "." << std::endl
          << AsciiDistributionCompare(sigma_draws_, sd_);

    }
    
  }  // namespace StateSpaceTesting
}  // namespace Trig

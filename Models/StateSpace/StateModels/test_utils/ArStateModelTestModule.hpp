#ifndef BOOM_STATE_SPACE_TEST_UTILS_AR_STATE_MODEL_TEST_MODULE_HPP_
#define BOOM_STATE_SPACE_TEST_UTILS_AR_STATE_MODEL_TEST_MODULE_HPP_
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

#include "Models/StateSpace/StateModels/ArStateModel.hpp"
#include "Models/TimeSeries/PosteriorSamplers/ArPosteriorSampler.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"

#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class ArStateModelTestModule
        : public StateModelTestModule<StateModel, ScalarStateSpaceModelBase> {
     public:
        ArStateModelTestModule(const Vector &ar_coefficients, double sd);
      
      void SimulateData(int time_dimension) override;
      
      Ptr<StateModel> get_state_model() override {
        return trend_model_;
      }

      void ObserveDraws(const ScalarStateSpaceModelBase &model) override {
        auto state = CurrentState(model);
        trend_draws_.row(cursor()) = state.row(0);
        sigma_draws_[cursor()] = trend_model_->sigma();
        coefficient_draws_.row(cursor()) = trend_model_->phi();
      }
      
      const Vector &StateContribution() const override { return trend_; }
      void CreateObservationSpace(int niter) override;
      void Check() override;

     private:
      Vector ar_coefficients_;
      double sd_;

      Ptr<ArStateModel> trend_model_;
      
      // The simulated state.
      Vector trend_;
      
      Matrix trend_draws_;
      Vector sigma_draws_;
      Matrix coefficient_draws_;
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM 

#endif // BOOM_STATE_SPACE_TEST_UTILS_AR_STATE_MODEL_TEST_MODULE_HPP_


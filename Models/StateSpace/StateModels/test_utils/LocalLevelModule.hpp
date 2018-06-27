#ifndef BOOM_STATE_SPACE_TEST_UTILS_LOCAL_LEVEL_HPP_
#define BOOM_STATE_SPACE_TEST_UTILS_LOCAL_LEVEL_HPP_
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

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class LocalLevelModule
        : public StateModelTestModule {
     public:
      LocalLevelModule(double level_sd, double initial_level);
      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override { return trend_; }
      void ImbueState(StateSpaceModelBase &model) override;
      void ImbueState(DynamicInterceptRegressionModel &model) override;
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const StateSpaceModelBase &model) override;
      void Check() override;
      
     private:
      double level_sd_;
      double initial_level_;
      Ptr<LocalLevelDynamicInterceptStateModel> trend_model_;
      Ptr<ChisqModel> level_precision_prior_;
      Ptr<ZeroMeanGaussianConjSampler> level_precision_sampler_;

      Vector trend_;
      int cursor_;
      // The index of this the state model in the state model being monitored.
      int state_model_index_;
      Matrix trend_draws_;
      Vector sigma_level_draws_;
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM 

#endif // BOOM_STATE_SPACE_TEST_UTILS_LOCAL_LEVEL_HPP_


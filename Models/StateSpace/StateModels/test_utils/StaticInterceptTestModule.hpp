#ifndef BOOM_STATE_SPACE_TEST_UTILS_STATIC_INTERCEPT_HPP_
#define BOOM_STATE_SPACE_TEST_UTILS_STATIC_INTERCEPT_HPP_
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

#include "Models/StateSpace/StateModels/StaticInterceptStateModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "Models/TimeSeries/NonzeroMeanAr1Model.hpp"
#include "Models/TimeSeries/PosteriorSamplers/NonzeroMeanAr1Sampler.hpp"

#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class StaticInterceptTestModule
        : public StateModelTestModule<StateModel, ScalarStateSpaceModelBase> {
     public:
      explicit StaticInterceptTestModule(double intercept);
      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override { return state_; }
      Ptr<StateModel> get_state_model() override {return intercept_model_;}
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const ScalarStateSpaceModelBase &model) override;
      void Check() override;
      
     private:
      double intercept_;
      Ptr<StaticInterceptStateModel> intercept_model_;
      // Ptr<DynamicInterceptStateModelAdapter> adapter_;

      Vector state_;
      Vector intercept_draws_;
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM 

#endif // BOOM_STATE_SPACE_TEST_UTILS_STATIC_INTERCEPT_HPP_


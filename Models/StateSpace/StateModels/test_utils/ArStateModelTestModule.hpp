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
        : public StateModelTestModule {
     public:
      ArStateModelTestModule(const Vector &ar_coefficients,
                             double sd);

      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override { return trend_; }
      Ptr<StateModel> get_state_model() override {return trend_model_;}
      Ptr<DynamicInterceptStateModel>
      get_dynamic_intercept_state_model() override {
        return trend_model_;
      }
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const StateSpaceModelBase &model) override;
      void Check() override;
      
     private:
      Vector ar_coefficients_;
      double sd_;

      Ptr<ArDynamicInterceptStateModel> trend_model_;
      Ptr<ChisqModel> precision_prior_;
      Ptr<ArPosteriorSampler> sampler_;

      // The simulated state.
      Vector trend_;

      int cursor_;
      Matrix trend_draws_;
      Vector sigma_draws_;
      Matrix coefficient_draws_;
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM 

#endif // BOOM_STATE_SPACE_TEST_UTILS_AR_STATE_MODEL_TEST_MODULE_HPP_


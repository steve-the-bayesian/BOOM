#ifndef BOOM_STATE_SPACE_SEASONAL_TEST_MODULE_HPP_
#define BOOM_STATE_SPACE_SEASONAL_TEST_MODULE_HPP_
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

#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class SeasonalTestModule
        : public StateModelTestModule<StateModel, ScalarStateSpaceModelBase> {
     public:
      // Args:
      //   sd:  The standard deviation of the state innovation errors.
      //   pattern: The initial seasonal pattern to use.  The elements of
      //     'pattern' are increasing in time, and describe a full cycle.  The
      //     mean of 'pattern' wlil be subtracted away to make the pattern have
      //     mean zero.  The length of pattern determines the number of seasons.
      //   season_duration:  The number of time periods each season lasts.
      SeasonalTestModule(double sd, const Vector &pattern, int season_duration = 1);

      // With this constructor, the initial state is simply random noise.
      SeasonalTestModule(double sd, int nseasons, int season_duration = 1);

      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override { return seasonal_; }
      Ptr<StateModel> get_state_model() override {return seasonal_model_;}
      // Ptr<DynamicInterceptStateModel>
      // get_dynamic_intercept_state_model() override { return adapter_; }
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const ScalarStateSpaceModelBase &model) override;
      void Check() override;

      // Given a seasonal pattern, convert it to state by subtracting the mean,
      // removing the first element, and reversing the time order.
      Vector pattern_to_state(const Vector &pattern) {
        Vector tmp = pattern - mean(pattern);
        return rev(ConstVectorView(tmp, 1));
      }

      Vector random_initial_pattern(int nseasons) {
        Vector ans(nseasons);
        ans.randomize();
        return ans;
      }
      
     private:
      double sd_;
      Vector initial_pattern_;
      int season_duration_;
      int state_dim_;
      Ptr<SeasonalStateModel> seasonal_model_;
      // Ptr<DynamicInterceptStateModelAdapter> adapter_;
      Ptr<ChisqModel> precision_prior_;
      Ptr<ZeroMeanGaussianConjSampler> sampler_;

      Vector seasonal_;
      Matrix seasonal_draws_;
      Vector sigma_draws_;
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_SEASONAL_TEST_MODULE_HPP_

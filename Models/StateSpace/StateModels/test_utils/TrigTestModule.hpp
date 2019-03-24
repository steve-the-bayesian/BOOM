#ifndef BOOM_STATE_SPACE_TRIG_TEST_MODULE_HPP_
#define BOOM_STATE_SPACE_TRIG_TEST_MODULE_HPP_
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

#include "Models/StateSpace/StateModels/TrigStateModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class TrigTestModule
        : public StateModelTestModule<StateModel, ScalarStateSpaceModelBase> {
     public:
      // Args:
      //   period: The number of time steps (need not be an integer) that it
      //     takes for the longest cycle to repeat.  E.g. 7 for a day of week
      //     cycle, or 12/52.178/365.25 for an annual cycle based on
      //     monthly/weekly/daily data.
      //   frequencies: A vector giving the number of times each sinusoid repeats
      //     in a period.  One sine and one cosine will be added to the model for
      //     each entry in frequencies.  This is typically {1, 2, ...}.
      //   sd:  The standard deviation of the state innovation errors.
      TrigTestModule(double period, const Vector &frequencies, double sd);

      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override { return trig_; }
      Ptr<StateModel> get_state_model() override {return trig_model_;}
      // Ptr<DynamicInterceptStateModel>
      // get_dynamic_intercept_state_model() override {return adapter_;}
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const ScalarStateSpaceModelBase &model) override;
      void Check() override;
      
     private:
      double period_;
      Vector frequencies_;
      double sd_;
      Ptr<TrigStateModel> trig_model_;
      // Ptr<DynamicInterceptStateModelAdapter> adapter_;

      // The prior for the error distribution.
      Ptr<ChisqModel> precision_prior_;
      Ptr<ZeroMeanGaussianConjSampler> sampler_;

      // The simulated data for the test.
      Vector trig_;

      // Space for observing MCMC draws.
      Matrix trig_draws_;
      Vector sigma_draws_;
    };
    
  }  // namespace StateSpaceTesting
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_TRIG_TEST_MODULE_HPP_

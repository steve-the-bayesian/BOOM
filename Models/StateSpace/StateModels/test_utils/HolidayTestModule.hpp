#ifndef BOOM_STATE_SPACE_HOLIDAY_TEST_MODULE_HPP_
#define BOOM_STATE_SPACE_HOLIDAY_TEST_MODULE_HPP_
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

#include "Models/StateSpace/StateModels/RandomWalkHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/RegressionHolidayStateModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class RandomWalkHolidayTestModule
        : public StateModelTestModule {
     public:
      // Args:
      //   holiday:  The holiday to be modeled.
      //   day_zero: The 'time zero' argument required by
      //     RegressionHolidayStateModel.
      //   sd: The standard deviation of the random walk innovations from one
      //     year to the next.
      //   initial_pattern: The holiday pattern at time zero.  The length of
      //     this vector must match the window width of the holiday.
      RandomWalkHolidayTestModule(const Ptr<Holiday> &holiday,
                                  const Date &day_zero,
                                  double sd,
                                  const Vector &initial_pattern);

      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override { return holiday_effect_; }
      Ptr<StateModel> get_state_model() override {return holiday_model_;}
      Ptr<DynamicInterceptStateModel>
      get_dynamic_intercept_state_model() override {
        return holiday_model_;
      }
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const StateSpaceModelBase &model) override;
      void Check() override;

     private:
      double sd_;
      Vector initial_pattern_;
      Date day_zero_;
      Ptr<Holiday> holiday_;
      
      Ptr<RandomWalkHolidayDynamicInterceptStateModel> holiday_model_;
      Ptr<ChisqModel> precision_prior_;
      Ptr<ZeroMeanGaussianConjSampler> sampler_;

      Vector holiday_effect_;
      Matrix holiday_draws_;
      Vector sd_draws_;
    };

    //==========================================================================
    class RegressionHolidayTestModule
        : public StateModelTestModule {
     public:
      RegressionHolidayTestModule(const Date &day_zero);

      void AddHoliday(const Ptr<Holiday> &holiday,
                      const Vector &pattern);
                                  
      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override { return holiday_effect_; }
      Ptr<StateModel> get_state_model() override {return scalar_holiday_model_;}
      Ptr<DynamicInterceptStateModel>
      get_dynamic_intercept_state_model() override {
        return dynamic_holiday_model_;
      }
      void ImbueState(ScalarStateSpaceModelBase &model) override;
      void ImbueState(DynamicInterceptRegressionModel &model) override;
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const StateSpaceModelBase &model) override;
      void Check() override;

     private:
      Date day_zero_;
      Ptr<GaussianModel> regression_coefficient_prior_;

      Ptr<RegressionHolidayStateModel> holiday_model_;
      Ptr<ScalarRegressionHolidayStateModel> scalar_holiday_model_;
      Ptr<DynamicInterceptRegressionHolidayStateModel> dynamic_holiday_model_;
      std::vector<Ptr<Holiday>> holidays_;
      std::vector<Vector> holiday_patterns_;
      
      Vector holiday_effect_;
      Matrix holiday_effect_draws_;
    };

  }  // namespace StateSpaceTesting
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_HOLIDAY_TEST_MODULE_HPP_

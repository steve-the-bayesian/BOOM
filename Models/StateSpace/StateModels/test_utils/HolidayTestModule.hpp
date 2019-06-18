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
#include "Models/StateSpace/StateModels/HierarchicalRegressionHolidayStateModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/GaussianModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/WishartModel.hpp"
#include "Models/Hierarchical/PosteriorSamplers/HierGaussianRegressionAsisSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/StateModels/test_utils/StateTestModule.hpp"

namespace BOOM {
  namespace StateSpaceTesting {

    class RandomWalkHolidayTestModule
        : public StateModelTestModule<StateModel, ScalarStateSpaceModelBase> {
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
      // Ptr<DynamicInterceptStateModel>
      // get_dynamic_intercept_state_model() override {
      //   return adapter_;
      // }
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const ScalarStateSpaceModelBase &model) override;
      void Check() override;

     private:
      double sd_;
      Vector initial_pattern_;
      Date day_zero_;
      Ptr<Holiday> holiday_;
      
      Ptr<RandomWalkHolidayStateModel> holiday_model_;
      // Ptr<DynamicInterceptStateModelAdapter> adapter_;
      Ptr<ChisqModel> precision_prior_;
      Ptr<ZeroMeanGaussianConjSampler> sampler_;

      Vector holiday_effect_;
      Matrix holiday_draws_;
      Vector sd_draws_;
    };

    //==========================================================================
    class RegressionHolidayTestModuleBase:
        public StateModelTestModule<StateModel, ScalarStateSpaceModelBase> {
     public:
      RegressionHolidayTestModuleBase(const Date &day_zero)
          : day_zero_(day_zero) {}

      // Add a holiday to the test module.  This will store the holiday in a
      // buffer where it can be used for fake data simuilation until it is time
      // to build the model.  Then AddHolidayToModel will be called to add the
      // holiday to the state model.
      virtual void AddHoliday(const Ptr<Holiday> &holiday, const Vector &pattern);

      void SimulateData(int time_dimension) override;
      const Vector &StateContribution() const override {
        return holiday_effect_;
      }

      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const ScalarStateSpaceModelBase &model) override;
      void Check() override;

      const Date &day_zero() const {return day_zero_;}
      
     protected:
      const std::vector<Ptr<Holiday>> holidays() const { return holidays_; }
      const std::vector<Vector> holiday_patterns() const {
        return holiday_patterns_;
      }
      
      void SetHolidays() {
        for (const auto &h : holidays()) {
          AddHolidayToModel(h);
        }
      }
      
     private:
      // Part of the implementation of AddHoliday
      virtual void AddHolidayToModel(const Ptr<Holiday> &holiday) = 0;
      virtual Ptr<StateModel> holiday_model() = 0;
      
      Date day_zero_;
      std::vector<Ptr<Holiday>> holidays_;
      std::vector<Vector> holiday_patterns_;
      
      Vector holiday_effect_;
      Matrix holiday_effect_draws_;
    };
    
    //==========================================================================
    class RegressionHolidayTestModule
        : public RegressionHolidayTestModuleBase {
     public:
      explicit RegressionHolidayTestModule(const Date &day_zero);

      Ptr<StateModel> get_state_model() override {
        return scalar_holiday_model_;
      }

      void ImbueState(ScalarStateSpaceModelBase &model) override;

     private:
      void AddHolidayToModel(const Ptr<Holiday> &h) override {
        holiday_model_->add_holiday(h);
      }

      Ptr<StateModel> holiday_model() override {
        return holiday_model_;
      }
      
      Date day_zero_;
      Ptr<GaussianModel> regression_coefficient_prior_;

      Ptr<RegressionHolidayStateModel> holiday_model_;
      Ptr<ScalarRegressionHolidayStateModel> scalar_holiday_model_;
      // Ptr<DynamicInterceptRegressionHolidayStateModel> dynamic_holiday_model_;
      std::vector<Ptr<Holiday>> holidays_;
      std::vector<Vector> holiday_patterns_;
    };
    //==========================================================================
    class HierarchicalRegressionHolidayTestModule
        : public RegressionHolidayTestModuleBase {
     public:
      explicit HierarchicalRegressionHolidayTestModule(const Date &day_zero);
      
      void AddHoliday(const Ptr<Holiday> &holiday, const Vector &pattern) override;

      Ptr<StateModel> get_state_model() override {
        return scalar_holiday_model_;
      }
      
      void ImbueState(ScalarStateSpaceModelBase &model) override;
      void CreateObservationSpace(int niter) override;
      void ObserveDraws(const ScalarStateSpaceModelBase &model) override;
      void Check() override;

     private:
      void SetPrior();
      void AddHolidayToModel(const Ptr<Holiday> &h) override {
        holiday_model_->add_holiday(h);
      }
      Ptr<StateModel> holiday_model() override {
        return holiday_model_;
      }
      
      Date day_zero_;

      // These two model pointers start off as nullptr.  A call to ImbueState
      // sets one of them to the model being tested.
      Ptr<ScalarHierarchicalRegressionHolidayStateModel> scalar_holiday_model_;
      // Ptr<DynamicInterceptHierarchicalRegressionHolidayStateModel>
      // dynamic_holiday_model_;

      // After ImbueState(), holiday_model_ points to one of the two models
      // above.
      Ptr<HierarchicalRegressionHolidayStateModel> holiday_model_;
      Ptr<HierGaussianRegressionAsisSampler> sampler_;

      // The hyperprior distribution.  
      Ptr<MvnModel> holiday_mean_prior_;
      Ptr<WishartModel> holiday_precision_prior_;
      
    };
  }  // namespace StateSpaceTesting
}  // namespace BOOM

#endif  //  BOOM_STATE_SPACE_HOLIDAY_TEST_MODULE_HPP_

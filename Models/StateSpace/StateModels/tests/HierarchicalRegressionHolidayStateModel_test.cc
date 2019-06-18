#include "gtest/gtest.h"
#include "Models/Hierarchical/PosteriorSamplers/HierGaussianRegressionAsisSampler.hpp"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "Models/StateSpace/StateModels/HierarchicalRegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"

#include "cpputil/Date.hpp"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/HolidayTestModule.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;
  
  class HierarchicalRegressionHolidayStateModelTest : public ::testing::Test {
   protected:
    HierarchicalRegressionHolidayStateModelTest()
        : t0_(May, 15, 2004),
          residual_variance_(new UnivParams(1.3)),
          prior_(new GaussianModel(0, 2))
    {
      GlobalRng::rng.seed(8675309);
    }

    Date t0_;
    Ptr<UnivParams> residual_variance_;
    Ptr<GaussianModel> prior_;
  };

  TEST_F(HierarchicalRegressionHolidayStateModelTest, EverythingTest) {
    // Simulate state and data.
    Vector y(365 * 2);
    Vector mu(y.size());
    Date date = t0_;

    int number_of_holidays = 8;
    Vector b0 = {.3, 1.5, -.2};
    Matrix patterns(number_of_holidays, 3);
    Date holiday_date = t0_ + 12;
    std::vector<Ptr<Holiday>> holidays;
    for (int i = 0; i < number_of_holidays; ++i) {
      for (int day = 0; day < 3; ++day) {
        patterns(i, day) = rnorm(b0[day], .2);
      }
      NEW(FixedDateHoliday, holiday)(
          holiday_date.month(),
          holiday_date.day());
      holidays.push_back(holiday);
      holiday_date += 30;
    }

    for (int i = 0; i < y.size(); ++i, ++date) {
      mu[i] = (i == 0) ? 0 : mu[i-1] + rnorm(0, .1);
      y[i] = mu[i] + rnorm(0, 1);
      for (int h = 0; h < number_of_holidays; ++h) {
        if (holidays[h]->active(date)) {
          int d = holidays[h]->days_into_influence_window(date);
          y[i] += patterns(h, d);
          break;
        }
      }
    }

    //----------------------------------------------------------------------
    // Build the models.
    StateSpaceModel model(y);
    Matrix state(2, y.size());
    state.row(0) = mu;
    state.row(1) = 1.0;

    NEW(LocalLevelStateModel, level)();
    level->set_initial_state_mean(y[0]);
    model.add_state(level);

    NEW(ScalarHierarchicalRegressionHolidayStateModel, holiday_model)(
        t0_, &model);

    NEW(MvnModel, holiday_mean_prior)(3);
    NEW(WishartModel, holiday_variance_prior)(3);
    for (int i = 0; i < holidays.size(); ++i) {
      holiday_model->add_holiday(holidays[i]);
    }
    holiday_model->observe_time_dimension(y.size());
    NEW(HierGaussianRegressionAsisSampler, holiday_sampler)(
        holiday_model->model(),
        holiday_mean_prior,
        holiday_variance_prior,
        nullptr);
    holiday_model->set_method(holiday_sampler);
    model.add_state(holiday_model);

    //----------------------------------------------------------------------    
    // Now test some stuff.
    EXPECT_DOUBLE_EQ(holiday_model->initial_state_mean()[0], 1.0);
    EXPECT_DOUBLE_EQ(holiday_model->initial_state_variance()(0, 0), 0.0);
    for (int h = 0; h < number_of_holidays; ++h) {
      holiday_model->model()->data_model(h)->set_Beta(patterns.row(h));
    }
    model.permanently_set_state(state);

    int first_holiday = 12;
    int second_holiday = first_holiday + 30;

    //----------------------------------------------------------------------
    // Check observe_state.
    
    // After observing the first day of the first holiday window, the upper left
    // element of X'X is 1, and the first element of X'y contains the residual.
    // Other elements are zero.
    holiday_model->observe_state(
        ConstVectorView(state.col(first_holiday - 1), 1, 1),
        ConstVectorView(state.col(first_holiday - 2), 1, 1),
        first_holiday - 1);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xtx()(0, 0),
                     1.0)
        << "X'X =" << endl
        << holiday_model->model()->data_model(0)->suf()->xtx();
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xty()[0],
                     y[first_holiday - 1] - mu[first_holiday - 1]);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xtx()(1, 1),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xty()[1],
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xtx()(2, 2),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xty()[2],
                     0.0);

    // After observing the second day of the first holiday, the first two
    // diagonal elements of X'X contain 1's, and the other elements are zero.
    // The first two residuals are in the first two elements of X'y.
    holiday_model->observe_state(
        ConstVectorView(state.col(first_holiday), 1, 1),
        ConstVectorView(state.col(first_holiday - 1), 1, 1),
        first_holiday);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xtx()(0, 0),
                     1.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xty()[0],
                     y[first_holiday - 1] - mu[first_holiday - 1]);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xtx()(1, 1),
                     1.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xty()[1],
                     y[first_holiday] - mu[first_holiday]);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xtx()(2, 2),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(0)->suf()->xty()[2],
                     0.0);

    // Check the second holiday, which is in data_model(1).
    holiday_model->observe_state(
        ConstVectorView(state.col(second_holiday), 1, 1),
        ConstVectorView(state.col(second_holiday - 1), 1, 1),
        second_holiday);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xtx()(0, 0),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xty()[0],
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xtx()(1, 1),
                     1.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xty()[1],
                     y[second_holiday] - mu[second_holiday]);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xtx()(2, 2),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xty()[2],
                     0.0);

    // Check clear_data
    holiday_model->clear_data();
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xtx()(0, 0),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xty()[0],
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xtx()(1, 1),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xty()[1],
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xtx()(2, 2),
                     0.0);
    EXPECT_DOUBLE_EQ(holiday_model->model()->data_model(1)->suf()->xty()[2],
                     0.0);

    // Check that the observation coefficients are as expected.
    EXPECT_DOUBLE_EQ(holiday_model->observation_matrix(first_holiday - 1)[0],
                     patterns(0, 0));
    EXPECT_DOUBLE_EQ(holiday_model->observation_matrix(first_holiday)[0],
                     patterns(0, 1));
    EXPECT_DOUBLE_EQ(holiday_model->observation_matrix(first_holiday + 1)[0],
                     patterns(0, 2));
    EXPECT_DOUBLE_EQ(holiday_model->observation_matrix(second_holiday-1)[0],
                     patterns(1, 0));
    EXPECT_DOUBLE_EQ(holiday_model->observation_matrix(second_holiday)[0],
                     patterns(1, 1));
    EXPECT_DOUBLE_EQ(holiday_model->observation_matrix(second_holiday + 1)[0],
                     patterns(1, 2));
  }

  TEST_F(HierarchicalRegressionHolidayStateModelTest, StateSpaceFramework) {
    StateSpaceTestFramework framework(1.2);
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules;
    HierarchicalRegressionHolidayTestModule *holiday_module(
        new HierarchicalRegressionHolidayTestModule(t0_));
    modules.AddModule(holiday_module);
    
    int number_of_holidays = 8;
    Vector b0 = {.3, 1.5, -.2};
    Date holiday_date = t0_ + 12;
    std::vector<Ptr<Holiday>> holidays;
    for (int i = 0; i < number_of_holidays; ++i) {
      Vector pattern(3);
      for (int day = 0; day < 3; ++day) {
        pattern(day) = rnorm(b0[day], .2);
      }
      NEW(FixedDateHoliday, holiday)(
          holiday_date.month(),
          holiday_date.day());
      holiday_module->AddHoliday(holiday, pattern);
      holiday_date += 30;
    }

    framework.AddState(modules);
    framework.Test(500, 3 * 365);
  }
  
}  // namespace

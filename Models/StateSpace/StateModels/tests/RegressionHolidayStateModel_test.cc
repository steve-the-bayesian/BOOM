#include "gtest/gtest.h"
#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "Models/StateSpace/StateModels/RegressionHolidayStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "cpputil/Date.hpp"
#include "distributions.hpp"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/HolidayTestModule.hpp"
#include "Models/StateSpace/StateModels/test_utils/LocalLevelModule.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;
  
  class RegressionHolidayStateModelTest : public ::testing::Test {
   protected:
    RegressionHolidayStateModelTest()
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

  TEST_F(RegressionHolidayStateModelTest, Impl) {
    RegressionHolidayBaseImpl impl(t0_, residual_variance_);

    EXPECT_EQ(-1, impl.which_holiday(3));
    EXPECT_EQ(-1, impl.which_day(3));

    NEW(FixedDateHoliday, May18)(May, 18);
    impl.add_holiday(May18);
    NEW(FixedDateHoliday, July4)(Jul, 4);
    impl.add_holiday(July4);

    impl.observe_time_dimension(365);
    // May 18 is 3 days after May 15... time zero
    EXPECT_TRUE(May18->active(t0_ + 2));
    EXPECT_EQ(0, impl.which_holiday(2));
    EXPECT_EQ(0, impl.which_day(2));
    
    EXPECT_TRUE(May18->active(t0_ + 3));
    EXPECT_EQ(0, impl.which_holiday(3));
    EXPECT_EQ(1, impl.which_day(3));

    EXPECT_TRUE(May18->active(t0_ + 4));
    EXPECT_EQ(0, impl.which_holiday(4));
    EXPECT_EQ(2, impl.which_day(4));

    EXPECT_FALSE(May18->active(t0_ + 5));
    EXPECT_EQ(-1, impl.which_holiday(5));
    EXPECT_EQ(-1, impl.which_day(5));

    EXPECT_DOUBLE_EQ(1.3,
                     impl.residual_variance_value());
    residual_variance_->set(2.7);
    EXPECT_DOUBLE_EQ(2.7,
                     impl.residual_variance_value());
  }

  TEST_F(RegressionHolidayStateModelTest, WithStateSpace) {
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules;
    modules.AddModule(new LocalLevelModule(.2, 0));
    
    RegressionHolidayTestModule *reg_holiday_module =
        new RegressionHolidayTestModule(t0_);
    NEW(FixedDateHoliday, May18)(May, 18);
    reg_holiday_module->AddHoliday(May18, Vector{-3, 4, 12});
    NEW(FixedDateHoliday, July4)(Jul, 4);
    reg_holiday_module->AddHoliday(July4, Vector{-5, 10, 2});
    modules.AddModule(reg_holiday_module);

    
    StateSpaceTestFramework framework(.3);
    framework.AddState(modules);
    framework.Test(500, 3 * 365);
  }
  
  TEST_F(RegressionHolidayStateModelTest, RHSM) {
    Vector y(365 * 2);
    Vector mu(y.size());
    Date date = t0_;
    for (int i = 0; i < y.size(); ++i, ++date) {
      mu[i] = (i == 0) ? 0 : mu[i-1] + rnorm(0, .1);
      y[i] = mu[i] + rnorm(0, 1);
      if (date == Date(May, 17, 2004) || date == Date(May, 17, 2005)) {
        y[i] += .25;
      } else if (date == Date(May, 18, 2004) || date == Date(May, 18, 2005)) {
        y[i] += .5;
      } else if (date == Date(May, 19, 2004) || date == Date(May, 19, 2005)) {
        y[i] -= .25;
      }

      if (date == Date(Jul, 3, 2004) || date == Date(Jul, 3, 2005)) {
        y[i] += .3;
      } else if (date == Date(Jul, 4, 2004) || date == Date(Jul, 4, 2005)) {
        y[i] += 1.25;
      } else if (date == Date(Jul, 5, 2004) || date == Date(Jul, 5, 2005)) {
        y[i] += .3;
      }
    }
    StateSpaceModel model(y);
    Matrix state(2, y.size());
    state.row(0) = mu;
    state.row(1) = 1.0;

    NEW(LocalLevelStateModel, level)();
    level->set_initial_state_mean(y[0]);
    model.add_state(level);

    NEW(ScalarRegressionHolidayStateModel, holiday_model)(
        t0_, &model, prior_);
    NEW(FixedDateHoliday, May18)(May, 18);
    holiday_model->add_holiday(May18);
    NEW(FixedDateHoliday, July4)(Jul, 4);
    holiday_model->add_holiday(July4);
    holiday_model->observe_time_dimension(y.size());
    holiday_model->set_holiday_pattern(0, Vector({.25, .5, -0.25}));
    holiday_model->set_holiday_pattern(1, Vector({.3, 1.25, .3}));
    EXPECT_TRUE(VectorEquals(holiday_model->holiday_pattern(0),
                             Vector({.25, .5, -0.25})));
    EXPECT_TRUE(VectorEquals(holiday_model->holiday_pattern(1),
                             Vector({.3, 1.25, .3})));

    model.add_state(holiday_model);

    EXPECT_DOUBLE_EQ(holiday_model->initial_state_mean()[0], 1.0);
    EXPECT_DOUBLE_EQ(holiday_model->initial_state_variance()(0, 0), 0.0);
    model.permanently_set_state(state);
    
    int may_17_2004 = Date(May, 17, 2004) - t0_;
    holiday_model->observe_state(
        ConstVectorView(state.col(may_17_2004), 1, 1),
        ConstVectorView(state.col(may_17_2004 - 1), 1, 1),
        may_17_2004);

    EXPECT_TRUE( VectorEquals(
        Vector({1, 0, 0}),
        holiday_model->daily_counts(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({y[may_17_2004] - mu[may_17_2004], 0, 0}),
        holiday_model->daily_totals(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_counts(1)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_totals(1)));

    int may_18_2004 = may_17_2004 + 1;
    holiday_model->observe_state(
        ConstVectorView(state.col(may_18_2004), 1, 1),
        ConstVectorView(state.col(may_18_2004 - 1), 1, 1),
        may_18_2004);
    EXPECT_TRUE( VectorEquals(
        Vector({1, 1, 0}),
        holiday_model->daily_counts(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({y[may_17_2004] - mu[may_17_2004], y[may_18_2004] - mu[may_18_2004], 0}),
        holiday_model->daily_totals(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_counts(1)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_totals(1)));


    int july_4_2005 = Date(Jul, 4, 2005) - t0_;
    holiday_model->observe_state(
        ConstVectorView(state.col(july_4_2005), 1, 1),
        ConstVectorView(state.col(july_4_2005 - 1), 1, 1),
        july_4_2005);
    EXPECT_TRUE( VectorEquals(
        Vector({1, 1, 0}),
        holiday_model->daily_counts(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({y[may_17_2004] - mu[may_17_2004], y[may_18_2004] - mu[may_18_2004], 0}),
        holiday_model->daily_totals(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 1, 0}),
        holiday_model->daily_counts(1)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, y[july_4_2005] - mu[july_4_2005], 0}),
        holiday_model->daily_totals(1)));

    model.clear_client_data();
    EXPECT_TRUE( VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_counts(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_totals(0)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_counts(1)));
    EXPECT_TRUE(VectorEquals(
        Vector({0, 0, 0}),
        holiday_model->daily_totals(1)));

    EXPECT_EQ(1, holiday_model->observation_matrix(12).size());
    EXPECT_DOUBLE_EQ(.25, holiday_model->observation_matrix(may_17_2004)[0]);
    EXPECT_DOUBLE_EQ(.5, holiday_model->observation_matrix(may_18_2004)[0]);
    EXPECT_DOUBLE_EQ(-.25, holiday_model->observation_matrix(may_18_2004 + 1)[0]);

    EXPECT_EQ(1, holiday_model->observation_matrix(july_4_2005).size());
    EXPECT_DOUBLE_EQ(.3, holiday_model->observation_matrix(july_4_2005 - 1)[0]);
    EXPECT_DOUBLE_EQ(1.25, holiday_model->observation_matrix(july_4_2005)[0]);
    EXPECT_DOUBLE_EQ(.3, holiday_model->observation_matrix(july_4_2005 + 1)[0]);
  }
  
}  // namespace

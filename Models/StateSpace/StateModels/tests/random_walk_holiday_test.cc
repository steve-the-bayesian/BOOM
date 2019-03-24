#include "gtest/gtest.h"
#
#include "Models/StateSpace/StateModels/test_utils/HolidayTestModule.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "cpputil/Date.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;
  
  class RandomWalkHolidayStateModelTest : public ::testing::Test {
   protected:
    RandomWalkHolidayStateModelTest()
        : day0_(May, 15, 2004),
          holiday_date_(day0_ + 12),
          initial_pattern_{12.0, 15.0, 8.0},
          holiday_(new FixedDateHoliday(
              holiday_date_.month(), holiday_date_.day())),
          sd_(0.2)
    {
      GlobalRng::rng.seed(8675309);
      modules_.AddModule(new RandomWalkHolidayTestModule(
          holiday_, day0_, sd_, initial_pattern_));
    }

    Date day0_;
    Date holiday_date_;
    Vector initial_pattern_;
    Ptr<Holiday> holiday_;
    double sd_;
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules_;
  };

  TEST_F(RandomWalkHolidayStateModelTest, ModelMatrices) {
    StateSpaceModel model;
    modules_.ImbueState(model);
    EXPECT_EQ(model.state_dimension(), holiday_->maximum_window_width());
    EXPECT_FALSE(holiday_->active(day0_ + 9));
    EXPECT_FALSE(holiday_->active(day0_ + 10));
    EXPECT_TRUE(holiday_->active(day0_ + 11));
    EXPECT_TRUE(holiday_->active(day0_ + 12));
    EXPECT_TRUE(holiday_->active(day0_ + 13));
    EXPECT_FALSE(holiday_->active(day0_ + 14));

    Matrix variance(3, 3, 0.0);
    variance(0, 0) = square(sd_);
    EXPECT_TRUE(MatrixEquals(variance, model.state_variance_matrix(10)->dense()))
        << "dense: " << endl
        << variance
        << "sparse: " << model.state_variance_matrix(10)->dense();

    variance(0, 0) = 0.0;
    variance(1, 1) = square(sd_);
    EXPECT_TRUE(MatrixEquals(variance, model.state_variance_matrix(11)->dense()))
        << "dense: " << endl
        << variance
        << "sparse: " << model.state_variance_matrix(11)->dense();

    variance(1, 1) = 0;
    variance(2, 2) = square(sd_);
    EXPECT_TRUE(MatrixEquals(variance, model.state_variance_matrix(12)->dense()))
        << "dense: " << endl
        << variance
        << "sparse: " << model.state_variance_matrix(12)->dense();

    variance(2, 2) = 0;
    EXPECT_TRUE(MatrixEquals(variance, model.state_variance_matrix(13)->dense()))
        << "dense: " << endl
        << variance
        << "sparse: " << model.state_variance_matrix(13)->dense();
    EXPECT_TRUE(MatrixEquals(variance, model.state_variance_matrix(14)->dense()))
        << "dense: " << endl
        << variance
        << "sparse: " << model.state_variance_matrix(14)->dense();
    EXPECT_TRUE(MatrixEquals(variance, model.state_variance_matrix(9)->dense()))
        << "dense: " << endl
        << variance
        << "sparse: " << model.state_variance_matrix(9)->dense();
    
    // The transition matrix is always the identity.
    Matrix transition(3, 3);
    transition.diag() = 1.0;
    EXPECT_TRUE(MatrixEquals(model.state_transition_matrix(3)->dense(),
                             transition));
    EXPECT_TRUE(MatrixEquals(model.state_transition_matrix(12)->dense(),
                             transition));
    
  }
  
  TEST_F(RandomWalkHolidayStateModelTest, StateSpaceTest) {
    StateSpaceTestFramework state_space(1.2);
    state_space.AddState(modules_);
    int niter = 200;
    int time_dimension = 2 * 365;
    state_space.Test(niter, time_dimension, 20);
  }    

}  // namespace

#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/LocalLinearTrendModule.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;
  
  class LocalLinearTrendStateModelTest : public ::testing::Test {
   protected:
    LocalLinearTrendStateModelTest()
        : time_dimension_(100)
    {
      GlobalRng::rng.seed(8675309);
      double initial_level = 0.0;
      double initial_slope = 0.0;
      double level_sd = 0.3;
      double slope_sd = 0.1;
      modules_.AddModule(new LocalLinearTrendModule(
          level_sd, initial_level, slope_sd, initial_slope));
    }
    int time_dimension_;
    StateModuleManager modules_;
  };

  //======================================================================
  TEST_F(LocalLinearTrendStateModelTest, StateSpaceModelTest) {
    int niter = 400;
    StateSpaceTestFramework state_space(1.3);
    state_space.AddState(std::move(modules_));
    state_space.Test(niter, time_dimension_);
  }
  //======================================================================
  TEST_F(LocalLinearTrendStateModelTest, DynamicInterceptRegressionModelTest) {
    int niter = 300;
    Vector true_beta = {-3.2, 17.4, 12};
    DynamicInterceptTestFramework framework(true_beta, 1.3, 3.0);
    framework.AddState(std::move(modules_));
    framework.Test(niter, time_dimension_);
  }
  
}  // namespace

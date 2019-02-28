#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/ArStateModelTestModule.hpp"
#include "Models/StateSpace/StateModels/test_utils/StaticInterceptTestModule.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;
  
  class StaticInterceptStateModelTest : public ::testing::Test {
   protected:
    StaticInterceptStateModelTest()
        : time_dimension_(200)
    {
      GlobalRng::rng.seed(8675309);
      Vector ar_coefficients = {.6, .2};
      modules_.AddModule(new StaticInterceptTestModule(3.6));
      modules_.AddModule(new ArStateModelTestModule(ar_coefficients, 0.4));
    }
    int time_dimension_;
    StateModuleManager modules_;
  };

  //======================================================================
  TEST_F(StaticInterceptStateModelTest, StateSpaceModelTest) {
    int niter = 400;
    StateSpaceTestFramework state_space(1.3);
    state_space.AddState(modules_);
    state_space.Test(niter, time_dimension_);
  }
  //======================================================================
  TEST_F(StaticInterceptStateModelTest, DynamicInterceptRegressionModelTest) {
    int niter = 400;
    Vector true_beta = {-3.2, 17.4, 12};
    DynamicInterceptTestFramework framework(true_beta, 1.3, 3.0);
    framework.AddState(modules_);
    framework.Test(niter, time_dimension_);
  }
  
}  // namespace

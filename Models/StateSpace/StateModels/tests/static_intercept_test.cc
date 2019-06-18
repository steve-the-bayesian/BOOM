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
    }
    
    int time_dimension_;
  };

  //======================================================================
  TEST_F(StaticInterceptStateModelTest, StateSpaceModelTest) {
    int niter = 400;
    Vector ar_coefficients = {.6, .2};
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules;
    modules.AddModule(new StaticInterceptTestModule(3.6));
    modules.AddModule(new ArStateModelTestModule(ar_coefficients, 0.4));
    StateSpaceTestFramework state_space(1.3);
    state_space.AddState(modules);
    state_space.Test(niter, time_dimension_);
  }
  
}  // namespace

#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/ArStateModelTestModule.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class ArStateModelTest : public ::testing::Test {
   protected:
    ArStateModelTest()
        : time_dimension_(100)
    {
      GlobalRng::rng.seed(8675309);
      Vector ar_coefficients = {.6, .2};
      modules_.AddModule(new ArStateModelTestModule(ar_coefficients, 0.4));
    }
    int time_dimension_;
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules_;
  };

  //======================================================================
  TEST_F(ArStateModelTest, StateSpaceModelTest) {
    int niter = 1000;
    StateSpaceTestFramework state_space(1.3);
    state_space.AddState(modules_);
    state_space.Test(niter, time_dimension_);
  }

}  // namespace

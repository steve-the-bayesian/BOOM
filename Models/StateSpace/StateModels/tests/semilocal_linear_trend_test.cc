#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/SemilocalLinearTrendTestModule.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;
  
  class SemilocalLinearTrendStateModelTest : public ::testing::Test {
   protected:
    SemilocalLinearTrendStateModelTest()
        : time_dimension_(100)
    {
      GlobalRng::rng.seed(8675309);
      double initial_level = 0.0;
      double initial_slope = 0.0;
      double level_sd = 0.3;
      double slope_sd = 0.1;
      double slope_mean = 3.2;
      double slope_ar = .65;
      modules_.AddModule(new SemilocalLinearTrendTestModule(
          level_sd, initial_level,
          slope_sd, initial_slope,
          slope_mean, slope_ar));
    }
    int time_dimension_;
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules_;
  };

  //======================================================================
  TEST_F(SemilocalLinearTrendStateModelTest, StateSpaceModelTest) {
    int niter = 400;
    StateSpaceTestFramework state_space(1.3);
    state_space.AddState(modules_);
    state_space.Test(niter, time_dimension_);
  }
  
}  // namespace

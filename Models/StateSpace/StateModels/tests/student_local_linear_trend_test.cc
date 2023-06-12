#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/StudentLocalLinearTrendTestModule.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class StudentLocalLinearTrendStateModelTest : public ::testing::Test {
   protected:
    StudentLocalLinearTrendStateModelTest()
        : time_dimension_(300)
    {
      GlobalRng::rng.seed(8675309);
      double initial_level = 0.0;
      double initial_slope = 0.0;
      double level_sd = 0.3;
      double slope_sd = 0.1;
      double nu_level = 2.0;
      double nu_slope = 10.0;
      modules_.AddModule(new StudentLocalLinearTrendTestModule(
          level_sd, initial_level, nu_level,
          slope_sd, initial_slope, nu_slope));
    }
    int time_dimension_;
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules_;
  };

  //======================================================================
  TEST_F(StudentLocalLinearTrendStateModelTest, StateSpaceModelTest) {
    int niter = 600;
    int burn = 100;
    StateSpaceTestFramework state_space(1.3);
    state_space.AddState(modules_);
    state_space.Test(niter, time_dimension_, burn);
  }

}  // namespace

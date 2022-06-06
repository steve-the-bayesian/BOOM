#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/ScalarStateModelAdapter.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"

#include "Models/StateSpace/StateModels/test_utils/LocalLevelModule.hpp"
#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class ScalarStateModelAdapterTest : public ::testing::Test {
   protected:
    ScalarStateModelAdapterTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  //======================================================================
  TEST_F(ScalarStateModelAdapterTest, SmokeTest) {
    NEW(LocalLevelStateModel, trend)();
    NEW(SeasonalStateModel, seasonal)(4);

    int xdim = 1;
    int nseries = 12;
    MultivariateStateSpaceRegressionModel host(xdim, nseries);
    ConditionallyIndependentScalarStateModelMultivariateAdapter state(
        &host, nseries);
    state.add_state(trend);
    state.add_state(seasonal);

    EXPECT_EQ(state.state_dimension(), 1 + 3);
    EXPECT_EQ(state.state_error_dimension(), 1 + 1);

    Selector all(nseries, true);
    Selector none(nseries, false);
    Selector some(nseries, false);
    some.add(3);
    some.add(7);
    auto Z = state.observation_coefficients(0, all);
    Z = state.observation_coefficients(0, some);
    Z = state.observation_coefficients(0, none);

    Matrix transition{{1, 0, 0, 0}, {0, -1, -1, -1}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    EXPECT_TRUE(MatrixEquals(state.state_transition_matrix(0)->dense(),
                             transition))
        << state.state_transition_matrix(0)->dense();
  }

}  // namespace

#include "gtest/gtest.h"

#include "Models/StateSpace/StateModelVector.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"

namespace {

  using namespace BOOM;
  using BOOM::StateSpaceUtils::StateModelVector;
  class StateModelVectorTest : public ::testing::Test {
   protected:
    StateModelVectorTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(StateModelVectorTest, Smoke) {
    StateModelVector<StateModel> models;

    EXPECT_EQ(models.size(), 0);
    NEW(LocalLinearTrendStateModel, trend)();
    NEW(SeasonalStateModel, seasonal)(6);

    models.add_state(trend);
    models.add_state(seasonal);
    EXPECT_EQ(models.size(), 2);

    models.clear();
    EXPECT_EQ(models.size(), 0);
    EXPECT_EQ(models.begin(), models.end());
  }

}  // namespace

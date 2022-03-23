#include "gtest/gtest.h"

#include "Models/StateSpace/StateModelVector.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"

#include "test_utils/test_utils.hpp"

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
    NEW(SeasonalStateModel, seasonal)(4);

    models.add_state(trend);
    models.add_state(seasonal);
    EXPECT_EQ(models.size(), 2);
    EXPECT_EQ(models.state_dimension(), 2 + 3);
    EXPECT_EQ(models.state_error_dimension(), 2 + 1);
    EXPECT_EQ(models.number_of_state_models(), 2);

    Vector state(5);
    state.randomize();
    EXPECT_TRUE(VectorEquals(models.state_component(state, 0),
                             VectorView(state, 0, 2)));
    EXPECT_TRUE(VectorEquals(models.state_component(state, 1),
                             VectorView(state, 2, 3)));

    SpdMatrix V(5);
    V.randomize();
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        EXPECT_DOUBLE_EQ(V(i,j),
                         models.state_error_variance_component(V, 0)(i, j));
      }
    }
    for (int i = 0; i < 1; ++i) {
      for (int j = 0; j < 1; ++j) {
        EXPECT_DOUBLE_EQ(V(2 + i, 2 + j),
                         models.state_error_variance_component(V, 1)(i, j));
      }
    }

    Matrix full_state(5, 20);
    full_state.randomize();
    auto view = models.full_state_subcomponent(full_state, 0);
    EXPECT_EQ(2, view.nrow());
    EXPECT_EQ(20, view.ncol());
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 20; ++j) {
        EXPECT_DOUBLE_EQ(view(i, j), full_state(i, j));
      }
    }

    auto view2 = models.full_state_subcomponent(full_state, 1);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 20; ++j) {
        EXPECT_DOUBLE_EQ(view2(i, j), full_state(2 + i, j));
      }
    }

    models.clear();
    EXPECT_EQ(models.size(), 0);
    EXPECT_EQ(models.begin(), models.end());
    EXPECT_EQ(models.state_dimension(), 0);
    EXPECT_EQ(models.state_error_dimension(), 0);
    EXPECT_EQ(models.number_of_state_models(), 0);
  }

}  // namespace

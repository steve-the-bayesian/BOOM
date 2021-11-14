#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/tests/DynamicInterceptTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/LocalLevelModule.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"


#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class LocalLevelStateModelTest : public ::testing::Test {
   protected:
    LocalLevelStateModelTest()
        : time_dimension_(100)
    {
      GlobalRng::rng.seed(8675309);
      double initial_level = 0.0;
      double level_sd = 0.3;
      modules_.AddModule(new LocalLevelModule(level_sd, initial_level));
    }
    int time_dimension_;
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules_;
  };

  //======================================================================
  TEST_F(LocalLevelStateModelTest, ModelMatrices) {
    LocalLevelStateModel model(4.0);
    EXPECT_DOUBLE_EQ(model.sigsq(), 16.0);
    model.set_sigsq(4.0);
    Matrix Id(1, 1, 1.0);
    EXPECT_TRUE(MatrixEquals(
        model.state_transition_matrix(0)->dense(),
        Id));
    EXPECT_TRUE(MatrixEquals(
        model.state_error_expander(0)->dense(),
        Id));

    SpdMatrix V(1, 4.0);
    EXPECT_TRUE(MatrixEquals(
        model.state_variance_matrix(0)->dense(),
        V));
    EXPECT_TRUE(MatrixEquals(
        model.state_error_variance(0)->dense(),
        V));
    EXPECT_EQ(1, model.observation_matrix(0).size());
    EXPECT_DOUBLE_EQ(1.0, model.observation_matrix(0)[0]);
  }

}  // namespace

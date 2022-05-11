#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"
#include "Models/StateSpace/StateModels/test_utils/LocalLevelModule.hpp"
#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class SharedLocalLevelTest : public ::testing::Test {
   protected:
    SharedLocalLevelTest()
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
  TEST_F(SharedLocalLevelTest, StateSpaceModelTest) {
    int niter = 200;
    StateSpaceTestFramework state_space(1.3);
    state_space.AddState(modules_);
    state_space.Test(niter, time_dimension_);
  }

  TEST_F(SharedLocalLevelTest, SharedModelTest) {
    int nseries = 12;
    int nfactors = 3;
    MultivariateStateSpaceRegressionModel model(1, nseries);
    NEW(ConditionallyIndependentSharedLocalLevelStateModel, state_model)(&model, nfactors, nseries);
    model.add_state(state_model);

    EXPECT_TRUE(MatrixEquals(
        state_model->state_transition_matrix(3)->dense(),
        SpdMatrix(nfactors, 1.0)));

    Selector observed(nseries, true);
    EXPECT_EQ(state_model->observation_coefficients(3, observed)->nrow(),
              nseries);
    EXPECT_EQ(state_model->observation_coefficients(3, observed)->ncol(),
              nfactors);
    Matrix Z = state_model->observation_coefficients(3, observed)->dense();
    std::cerr << "Z = \n" << Z;
    // The observation coefficients should start off as an upper triangular
    // matrix of 1's.
    EXPECT_DOUBLE_EQ(Z(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(Z(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(Z(2, 2), 1.0);
    EXPECT_DOUBLE_EQ(Z(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(Z(0, 2), 1.0);
    EXPECT_DOUBLE_EQ(Z(1, 2), 1.0);
  }

}  // namespace

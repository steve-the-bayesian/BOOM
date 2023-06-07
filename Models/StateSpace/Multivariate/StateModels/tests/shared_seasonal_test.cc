#include "gtest/gtest.h"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedSeasonal.hpp"
#include "Models/StateSpace/StateModels/test_utils/SeasonalTestModule.hpp"
#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class SharedSeasonalTest : public ::testing::Test {
   protected:
    SharedSeasonalTest()
        : time_dimension_(100),
          pattern_{-1.0, 2.0, -1.0, 0.0}
    {
      GlobalRng::rng.seed(8675309);
    }
    int time_dimension_;
    Vector pattern_;
  };

//======================================================================
  TEST_F(SharedSeasonalTest, SharedModelTest) {
    int nseries = 12;
    int nfactors = 3;
    int nseasons = pattern_.size() - 1;
    int season_duration = 1;
    MultivariateStateSpaceRegressionModel model(1, nseries);
    NEW(SharedSeasonalStateModel, state_model)(
        &model, nfactors, nseasons, season_duration);
    model.add_state(state_model);

    int state_dimension = (nseasons - 1) * nfactors;
    EXPECT_EQ(state_model->state_dimension(), state_dimension);
    EXPECT_EQ(state_model->state_error_dimension(), nfactors);
    EXPECT_EQ(state_model->nseries(), nseries);
    EXPECT_EQ(state_model->number_of_factors(), nfactors);

    // With season_duration == 1 every time point is the start of a new season.
    for (int i = 0; i < 10; ++i) {
      EXPECT_TRUE(state_model->new_season(i));
    }

    // Examine the structural matrices.  The transition matrix is a block
    // diagonal matrix formed by 'nfactors' identical seasonal transition
    // matrices.
    Matrix dense_transition =
        state_model->state_transition_matrix(nfactors)->dense();
    SeasonalStateSpaceMatrix single_state_transition(nseasons);
    Matrix transition_target = block_diagonal(std::vector<Matrix>(
        nfactors, single_state_transition.dense()));
    EXPECT_TRUE(MatrixEquals(dense_transition, transition_target));

    // The state error variance is an identify matirx with dimension matching
    // 'nfactors'.
    SpdMatrix Id3(nfactors, 1.0);
    EXPECT_TRUE(MatrixEquals(
        state_model->state_error_variance(7)->dense(),
        Id3));

    // The state variance matrix is mostly zeros. There's a 1 at the start of
    // each factor block (corresponding to the current seasonal effect).  Factor
    // blocks are nseasons - 1 apart.
    SpdMatrix dense_state_variance(state_dimension, 0.0);
    for (int i = 0; i < state_dimension; i += nseasons - 1) {
      dense_state_variance(i, i) = 1.0;
    }
    EXPECT_TRUE(MatrixEquals(
        state_model->state_variance_matrix(5)->dense(),
        dense_state_variance));

    // The state error expander is a block diagonal matrix where each block is a
    // "tall skinny" matrix with a 1 in the first position.
    Matrix base_state_error_expander(nseasons - 1, 1, 0.0);
    base_state_error_expander(0, 0) = 1.0;
    EXPECT_TRUE(MatrixEquals(
        state_model->state_error_expander(5)->dense(),
        block_diagonal(std::vector<Matrix>(
            nfactors, base_state_error_expander))));

    // The observation coefficients are vectors of GlmCoefs, one per series,
    // relating to the full state.  We keep compressed GlmCoefs in the object.
    Vector ones(nfactors, 1.0);
    for (int i = 0; i < nseries; ++i) {
      EXPECT_TRUE(VectorEquals(
          ones, state_model->compressed_observation_coefficients(i)->Beta()));
    }

    // Test the observation coefficients.
    Selector fully_observed(nseries, true);
    Matrix observation_coefficients =
        state_model->observation_coefficients(3, fully_observed)->dense();
    EXPECT_EQ(observation_coefficients.nrow(), nseries);
    EXPECT_EQ(observation_coefficients.ncol(), state_dimension);
    Vector ones_ns(nseries, 1.0);
    Vector zeros_ns(nseries, 0.0);
    for (int i = 0; i < state_dimension; ++i) {
      if (i % (nseasons - 1) == 0) {
        EXPECT_TRUE(VectorEquals(ones_ns, observation_coefficients.col(i)));
      } else {
        EXPECT_TRUE(VectorEquals(zeros_ns, observation_coefficients.col(i)));
      }
    }
  }

}  // namespace

#include "gtest/gtest.h"
#include "distributions.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class KalmanFilterTest : public ::testing::Test {
   protected:
    KalmanFilterTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(KalmanFilterTest, ScalarFilter) {

    double true_sigma_level = .1;
    Vector trend(12, 0.0);
    for (int i = 1; i < trend.size(); ++i) {
      trend[i] = trend[i - 1] + rnorm(0, true_sigma_level);
    }

    double true_sigma_seasonal = .25;
    Matrix seasonal_transition(
        "-1 -1 -1|"
        "1   0  0|"
        "0   1  0");
    Vector seasonal_pattern = {-2, -1, 1};
    Vector seasonal_effect(12);
    for (int i = 0; i < 12; ++i) {
      seasonal_pattern = seasonal_transition * seasonal_pattern;
      seasonal_pattern[0] += rnorm(0, true_sigma_seasonal);
      seasonal_effect[i] = seasonal_pattern[0];
      if (i > 0) {
        EXPECT_DOUBLE_EQ(seasonal_effect[i-1], seasonal_pattern[1]);
      }
      if (i > 1) {
        EXPECT_DOUBLE_EQ(seasonal_effect[i-2], seasonal_pattern[2]);
      }
    }

    double true_sigma_obs = 1.3;
    Vector data = trend + seasonal_effect + rnorm_vector(12, 0, true_sigma_obs);

    NEW(LocalLevelStateModel, level)(square(true_sigma_level));
    NEW(SeasonalStateModel, seasonal)(4, 1);
    NEW(StateSpaceModel, model)();
    model->add_state(level);
    model->add_state(seasonal);
    model->observation_model()->set_sigsq(square(true_sigma_obs));

    Matrix level_transition(1, 1, 1);
    Matrix transition = block_diagonal(level_transition, seasonal_transition);

    EXPECT_TRUE(MatrixEquals(
        transition,
        model->state_transition_matrix(3)->dense()));

    // ScalarKalmanFilter &filter(model->get_filter());

    // TODO(finish this later)
  }

}  // namespace

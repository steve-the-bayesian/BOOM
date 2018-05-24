#include "gtest/gtest.h"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class StateSpaceModelTest : public ::testing::Test {
   protected:
    StateSpaceModelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // Checks to see that the one-step prediction errors are properly scaled.
  TEST_F(StateSpaceModelTest, PredictionErrors) {
    ifstream datafile("./Models/StateSpace/tests/airpassengers.txt");
    Vector y(datafile);
    ASSERT_EQ(y.size(), 144);

    Ptr<LocalLinearTrendStateModel> trend(new LocalLinearTrendStateModel);
    trend->set_initial_state_mean(Vector{y[0], 0});
    trend->set_initial_state_variance(SpdMatrix(2, 3.0));
    NEW(ZeroMeanMvnIndependenceSampler, trend_level_sampler)(
        trend.get(), 1, 1, 0);
    NEW(ZeroMeanMvnIndependenceSampler, trend_slope_sampler)(
        trend.get(), 1, 1, 1);
    trend->set_method(trend_level_sampler);
    trend->set_method(trend_slope_sampler);
    
    NEW(SeasonalStateModel, seasonal)(12);
    NEW(ZeroMeanGaussianConjSampler, seasonal_sampler)(
        seasonal.get(), 1, 1);
    seasonal->set_method(seasonal_sampler);
    seasonal->set_initial_state_mean(Vector(11, 0.0));
    seasonal->set_initial_state_variance(SpdMatrix(11, 3.0));
    
    NEW(StateSpaceModel, model)(log(y));
    model->add_state(trend);
    model->add_state(seasonal);

    NEW(ZeroMeanGaussianConjSampler, observation_model_sampler)(
        model->observation_model(), 1, 1);
    model->observation_model()->set_method(observation_model_sampler);

    NEW(StateSpacePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);
    
    for (int i = 0; i < 50; ++i) {
      model->sample_posterior();
    }

    Vector raw_errors = model->one_step_prediction_errors(false);
    EXPECT_EQ(raw_errors.size(), y.size());
    Vector variances = model->observation_error_variances();
    EXPECT_EQ(variances.size(), y.size());

    // Check that at least one of the variances is not 1.0.
    EXPECT_NE(variances[3], 1.0);
    Vector scaled_errors = model->one_step_prediction_errors(true);
    EXPECT_EQ(scaled_errors.size(), y.size());

    // Check that the raw and scaled errors are not the same thing.
    EXPECT_GE((raw_errors - scaled_errors).max_abs(), .05);
    // Check that you get to the scaled errors by dividing the raw ones by the
    // forecast SD.
    EXPECT_TRUE(VectorEquals(raw_errors / sqrt(variances), scaled_errors));
  }
  
}  // namespace

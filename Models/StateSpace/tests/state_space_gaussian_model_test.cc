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

    Ptr<StateSpaceModel> model_;
    Vector y_ = {
      112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
      115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
      145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
      171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
      196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
      204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
      242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
      284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
      315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
      340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
      360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
      417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
    };

    void setup() {
      //ifstream datafile("./Models/StateSpace/tests/airpassengers.txt");
      // y_.read(datafile);
      ASSERT_EQ(y_.size(), 144);

      Ptr<LocalLinearTrendStateModel> trend(new LocalLinearTrendStateModel);
      trend->set_initial_state_mean(Vector{log(y_[0]), 0});
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

      model_.reset(new StateSpaceModel(log(y_)));
      model_->add_state(trend);
      model_->add_state(seasonal);

      NEW(ZeroMeanGaussianConjSampler, observation_model_sampler)(
          model_->observation_model(), 1, 1);
      model_->observation_model()->set_method(observation_model_sampler);

      NEW(StateSpacePosteriorSampler, sampler)(model_.get());
      model_->set_method(sampler);
    }
  };

  // Checks to see that the one-step prediction errors are properly scaled.
  TEST_F(StateSpaceModelTest, PredictionErrors) {
    setup();
    for (int i = 0; i < 50; ++i) {
      model_->sample_posterior();
    }

    Vector raw_errors = model_->one_step_prediction_errors(false);
    EXPECT_EQ(raw_errors.size(), y_.size());
    Vector variances = model_->observation_error_variances();
    EXPECT_EQ(variances.size(), y_.size());

    // Check that at least one of the variances is not 1.0.
    EXPECT_NE(variances[3], 1.0);
    Vector scaled_errors = model_->one_step_prediction_errors(true);
    EXPECT_EQ(scaled_errors.size(), y_.size());

    // Check that the raw and scaled errors are not the same thing.
    EXPECT_GE((raw_errors - scaled_errors).max_abs(), .05);
    // Check that you get to the scaled errors by dividing the raw ones by the
    // forecast SD.
    EXPECT_TRUE(VectorEquals(raw_errors / sqrt(variances), scaled_errors));
  }

  TEST_F(StateSpaceModelTest, HoldoutPredictionErrors) {
    setup();
    model_->sample_posterior();

    using ::BOOM::StateSpaceUtils::compute_prediction_errors;
    std::vector<Matrix> errors = compute_prediction_errors(
        *model_,
        10,
        {60, 80, 90},
        false);
  }

}  // namespace

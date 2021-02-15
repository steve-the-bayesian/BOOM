#include "gtest/gtest.h"

#include "Models/StateSpace/StateSpaceRegressionModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"

#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionSemiconjugateSampler.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  class StateSpaceRegressionModelTest : public ::testing::Test {
   protected:
    StateSpaceRegressionModelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(StateSpaceRegressionModelTest, Forecasting) {
    int sample_size = 500;
    int train = 480;
    int xdim = 3;
    double innovation_sd = 1.1;
    double residual_sd = .2;

    NEW(StateSpaceRegressionModel, model)(xdim);

    //--------------------------------------------------------------------------
    // Simulate the data.
    //--------------------------------------------------------------------------
    Matrix predictors(sample_size, xdim);
    predictors.randomize();

    // Make the coefficients really obvious.
    Vector coefficients(xdim);
    for (int i = 0; i < xdim; ++i) coefficients[i] = (i + 1) * 10;
    Vector regression = predictors * coefficients;
    Vector state = cumsum(rnorm_vector(sample_size, 0, innovation_sd));
    Vector errors = rnorm_vector(sample_size, 0, residual_sd);

    Vector y = state + regression + errors;

    //--------------------------------------------------------------------------
    // Add a state model
    //--------------------------------------------------------------------------
    NEW(LocalLevelStateModel, state_model)(square(innovation_sd));
    NEW(ZeroMeanGaussianConjSampler, state_model_sampler)(
        state_model.get(), 1, innovation_sd);
    state_model->set_method(state_model_sampler);
    state_model->set_initial_state_mean(0);
    state_model->set_initial_state_variance(square(innovation_sd));
    model->add_state(state_model);

    EXPECT_EQ(1, model->number_of_state_models());

    //--------------------------------------------------------------------------
    // Add a posterior sampler for the observation model.
    //--------------------------------------------------------------------------
    NEW(RegressionSemiconjugateSampler, observation_model_sampler)(
        model->observation_model(),
        new MvnModel(Vector(xdim, 0), SpdMatrix(xdim, square(xdim * 100.0))),
        new ChisqModel(1, residual_sd));
    model->observation_model()->set_method(observation_model_sampler);

    //--------------------------------------------------------------------------
    // Add the sampler for the global model
    //--------------------------------------------------------------------------
    NEW(StateSpacePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    //--------------------------------------------------------------------------
    // Assign the data.
    //--------------------------------------------------------------------------
    for (int i = 0; i < train; ++i) {
      model->add_regression_data(new RegressionData(y[i], predictors.row(i)));
    }

    int burn = 200;
    for (int i = 0; i < burn; ++i) {
      model->sample_posterior();
    }

    int niter = 500;
    Matrix coefficient_draws(niter, xdim);
    Vector residual_sd_draws(niter);
    Matrix state_draws(niter, train);

    Matrix prediction_draws(niter, sample_size - train);
    Matrix test_predictors = ConstSubMatrix(
        predictors, train, nrow(predictors) - 1,
        0, ncol(predictors) - 1).to_matrix();

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      coefficient_draws.row(i) = model->observation_model()->Beta();
      residual_sd_draws[i] = model->observation_model()->sigma();
      state_draws.row(i) = model->state().row(0);

      prediction_draws.row(i) = model->simulate_forecast(GlobalRng::rng, test_predictors);
    }

    auto status = CheckMcmcMatrix(coefficient_draws, coefficients);
    EXPECT_TRUE(status.ok) << status;
    EXPECT_TRUE(CheckMcmcVector(residual_sd_draws, residual_sd));
    EXPECT_EQ("", CheckStochasticProcess(prediction_draws,
                                         ConstVectorView(y, train),
                                         .95, .2));
  }

}  // namespace

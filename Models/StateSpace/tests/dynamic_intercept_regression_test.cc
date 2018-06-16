#include "gtest/gtest.h"

#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include "Models/StateSpace/tests/state_space_test_utils.hpp"

#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class DynamicInterceptRegressionModelTest : public ::testing::Test {
   protected:
    DynamicInterceptRegressionModelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(DynamicInterceptRegressionModelTest, MCMC) {
    int time_dimension = 20;
    double true_level_sd = .3;
    Vector level = SimulateLocalLevel(
        GlobalRng::rng, time_dimension, 1.3, true_level_sd);

    double true_seasonal_sd = .15;
    Vector seasonal_pattern = {3, -1, 2};
    Vector seasonal = SimulateSeasonal(
        GlobalRng::rng, time_dimension, seasonal_pattern, true_seasonal_sd);


    
    Vector true_beta = {3, -2.4, 1.7};
    double true_observation_sd = 1.4;
    NEW(DynamicInterceptRegressionModel, model)(true_beta.size());
    
    for (int i = 0; i < time_dimension; ++i) {
      int nobs = 1 + rpois(2.0);
      Matrix predictors(nobs, true_beta.size());
      predictors.randomize();
      Vector response = predictors * true_beta;
      response += level[i] + seasonal[i];
      for (int j = 0; j < response.size(); ++j) {
        response[j] += rnorm(0, true_observation_sd);
      }
      NEW(StateSpace::TimeSeriesRegressionData, data_point)(
          response, predictors);
      model->add_data(data_point);
    }

    NEW(LocalLevelStateModel, level_model)(1.0);
    level_model->set_initial_state_mean(level[0]);
    level_model->set_initial_state_variance(var(level));
    NEW(ChisqModel, level_precision_prior)(1.0, square(true_level_sd));
    NEW(ZeroMeanGaussianConjSampler, level_sampler)(
        level_model.get(), level_precision_prior);
    level_model->set_method(level_sampler);
    model->add_state(level_model);

    NEW(SeasonalStateModel, seasonal_model)(1 + seasonal_pattern.size(), 1);
    seasonal_model->set_initial_state_mean(Vector(seasonal_pattern.size(), 0.0));
    seasonal_model->set_initial_state_variance(SpdMatrix(
        seasonal_pattern.size(), var(seasonal)));
    NEW(ChisqModel, seasonal_precision_prior)(1.0, square(true_seasonal_sd));
    NEW(ZeroMeanGaussianConjSampler, seasonal_sampler)(
        seasonal_model.get(), seasonal_precision_prior);
    seasonal_model->set_method(seasonal_sampler);
    model->add_state(seasonal_model);

    NEW(MvnGivenScalarSigma, coefficient_prior)(
        SpdMatrix(true_beta.size(), .01),
        model->observation_model()->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, square(true_observation_sd));
    NEW(VariableSelectionPrior, spike)(Vector{.999, .999, .999});
    NEW(BregVsSampler, regression_sampler)(
        model->observation_model(),
        coefficient_prior,
        residual_precision_prior,
        spike);
    model->observation_model()->set_method(regression_sampler);

    NEW(StateSpacePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    int niter = 200;
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
    }
    
  }
  
}  // namespace

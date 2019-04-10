#include "gtest/gtest.h"

#include "test_utils/test_utils.hpp"

#include "cpputil/math_utils.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionSemiconjugateSampler.hpp"
#include "Models/StateSpace/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"
#include "distributions.hpp"
#include "LinAlg/Array.hpp"

namespace {

  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultivariateStateSpaceRegressionModelTest : public ::testing::Test {
   protected:
    MultivariateStateSpaceRegressionModelTest()
    {
      GlobalRng::rng.seed(8675310);
    }
  };

  //===========================================================================
  TEST_F(MultivariateStateSpaceRegressionModelTest, EmptyTest) {}

  //===========================================================================
  TEST_F(MultivariateStateSpaceRegressionModelTest, ConstructorTest) {
    MultivariateStateSpaceRegressionModel model(3, 4);
  }

  TEST_F(MultivariateStateSpaceRegressionModelTest, DataTest) {
    TimeSeriesRegressionData data_point(3.2, Vector{1, 2, 3}, 0, 4);
    EXPECT_DOUBLE_EQ(3.2, data_point.y());
    EXPECT_TRUE(VectorEquals(Vector{1, 2, 3}, data_point.x()));
    EXPECT_EQ(0, data_point.series());
    EXPECT_EQ(4, data_point.timestamp());
  }

  TEST_F(MultivariateStateSpaceRegressionModelTest, ModelTest) {
    int ydim = 4;
    int xdim = 3;
        
    MultivariateStateSpaceRegressionModel model(xdim, ydim);
    EXPECT_EQ(0, model.state_dimension());
    EXPECT_EQ(0, model.number_of_state_models());
    EXPECT_EQ(nullptr, model.state_model(0));
    EXPECT_EQ(nullptr, model.state_model(-1));
    EXPECT_EQ(nullptr, model.state_model(2));
    EXPECT_EQ(0, model.time_dimension());

    EXPECT_EQ(ydim, model.nseries());
    EXPECT_EQ(xdim, model.xdim());

    std::vector<Ptr<TimeSeriesRegressionData>> data;
    Matrix response_data(ydim, 12);
    for (int time = 0; time < 12; ++time) {
      for (int series = 0; series < ydim; ++series){
        NEW(TimeSeriesRegressionData, data_point)(
            rnorm(0, 1), rnorm_vector(xdim, 0, 1), series, time);
        data.push_back(data_point);
        model.add_data(data_point);
        response_data(series, time) = data_point->y();
      }
    }
    EXPECT_EQ(12, model.time_dimension());
    for (int time = 0; time < 12; ++time) {
      for (int series = 0; series < ydim; ++series) {
        EXPECT_TRUE(model.is_observed(series, time));
        EXPECT_DOUBLE_EQ(response_data(series, time),
                         model.observed_data(series, time));
      }
    }
  }

  TEST_F(MultivariateStateSpaceRegressionModelTest, McmcTest) {
    // Simulate fake data from the model: shared local level and a regression
    // effect.

    int xdim = 3;
    int ydim = 6;
    int nfactors = 2;
    int sample_size = 100;
    double factor_sd = .3;
    double residual_sd = .1;

    //----------------------------------------------------------------------
    // Simulate the state.
    Matrix state(nfactors, sample_size);
    for (int factor = 0; factor < nfactors; ++factor) {
      state(factor, 0) = rnorm();
      for (int time = 1; time < sample_size; ++time) {
        state(factor, time) = state(factor, time - 1) + rnorm(0, factor_sd);
      }
    }

    // Set up the observation coefficients, which are zero above the diagonal
    // and 1 on the diagonal.
    Matrix observation_coefficients(ydim, nfactors);
    observation_coefficients.randomize();
    for (int i = 0; i < nfactors; ++i) {
      observation_coefficients(i, i) = 1.0;
      for (int j = i + 1; j < nfactors; ++j) {
        observation_coefficients(i, j) = 0.0;
      }
    }

    // Set up the regression coefficients and the predictors.
    Matrix regression_coefficients(ydim, xdim);
    regression_coefficients.randomize();
    Matrix predictors(sample_size, xdim);
    predictors.randomize();

    // Simulate the response.
    Matrix response(sample_size, ydim);
    for (int i = 0; i < sample_size; ++i) {
      Vector yhat = observation_coefficients * state.col(i)
          + regression_coefficients * predictors.row(i);
      for (int j = 0; j < ydim; ++j) {
        response(i, j) = yhat[j] + rnorm(0, residual_sd);
      }
    }

    //----------------------------------------------------------------------
    // Define the model.
    NEW(MultivariateStateSpaceRegressionModel, model)(xdim, ydim);
    for (int time = 0; time < sample_size; ++time) {
      for (int series = 0; series < ydim; ++series) {
        NEW(TimeSeriesRegressionData, data_point)(
            response(time, series), predictors.row(time), series, time);
        model->add_data(data_point);
      }
    }

    //---------------------------------------------------------------------------
    // Define the state model.
    NEW(SharedLocalLevelStateModel, state_model)(
        nfactors, model.get(), ydim);
    std::vector<Ptr<GammaModelBase>> innovation_precision_priors;
    for (int factor = 0; factor < nfactors; ++factor) {
      innovation_precision_priors.push_back(new ChisqModel(1.0, .10));
    }
    Matrix observation_coefficient_prior_mean(ydim, nfactors, 0.0);
    NEW(SharedLocalLevelPosteriorSampler, state_model_sampler)(
        state_model.get(),
        innovation_precision_priors,
        observation_coefficient_prior_mean,
        .01);
    state_model->set_method(state_model_sampler);
    model->add_state(state_model);

    //---------------------------------------------------------------------------
    // Set the prior for the regression model.
    for (int i = 0; i < ydim; ++i) {
      Vector beta_prior_mean(xdim, 0.0);
      SpdMatrix beta_precision(xdim, 1.0);
      NEW(MvnModel, beta_prior)(beta_prior_mean, beta_precision, true);
      NEW(ChisqModel, residual_precision_prior)(1.0, residual_sd);
      NEW(RegressionSemiconjugateSampler, regression_sampler)(
          model->observation_model()->model(i).get(),
          beta_prior, residual_precision_prior);
      model->observation_model()->model(i)->set_method(regression_sampler);
    }


    
    
  }
  
}  // namespace

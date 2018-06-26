#include "gtest/gtest.h"

#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnIndependenceSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"

#include "cpputil/Constants.hpp"
#include "cpputil/Date.hpp"
#include "cpputil/seq.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "test_utils/test_utils.hpp"

#include "Models/StateSpace/tests/state_space_test_utils.hpp"

#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class LocalLinearTrendStateModelTest : public ::testing::Test {
   protected:
    LocalLinearTrendStateModelTest()
        : sample_size_(100),
          xdim_(3),
          true_level_sd_(.3),
          true_slope_sd_(.1),
          true_observation_sd_(1.4)
    {
      GlobalRng::rng.seed(8675309);
    }
    int sample_size_;
    int xdim_;
    double true_level_sd_;
    double true_slope_sd_;
    double true_observation_sd_;
    Ptr<ChisqModel> level_precision_prior_;
    Ptr<ChisqModel> slope_precision_prior_;
    Ptr<ChisqModel> observation_precision_prior_;
    Vector trend_;
    Vector data_;
    Ptr<LocalLinearTrendStateModel> trend_model_;

    void SimulateData() {
      trend_ = SimulateLocalLinearTrend(
          GlobalRng::rng, sample_size_, 1.3, .1, true_level_sd_,
          true_slope_sd_);
      data_ = trend_;
      for (int i = 0; i < data_.size(); ++i) {
        data_[i] += rnorm_mt(GlobalRng::rng, 0, true_observation_sd_);
      }
    }

    void BuildModel() {
      trend_model_.reset(new LocalLinearTrendStateModel);
      level_precision_prior_.reset(new ChisqModel(1.0, true_level_sd_));
      NEW(ZeroMeanMvnIndependenceSampler, level_sampler)(
          trend_model_.get(), level_precision_prior_, 0);
      trend_model_->set_method(level_sampler);

      slope_precision_prior_.reset(new ChisqModel(1.0, true_slope_sd_));
      NEW(ZeroMeanMvnIndependenceSampler, slope_sampler)(
          trend_model_.get(), slope_precision_prior_, 1);
      trend_model_->set_method(slope_sampler);
      trend_model_->set_initial_state_mean(
          Vector{data_[0], data_[1] - data_[0]});
      trend_model_->set_initial_state_variance(SpdMatrix(2, var(data_)));
          
      observation_precision_prior_.reset(new ChisqModel(
          1.0, true_observation_sd_));
    }

  };

  //======================================================================
  TEST_F(LocalLinearTrendStateModelTest, StateSpaceModelTest) {
    SimulateData();
    BuildModel();

    StateSpaceModel model(data_);
    model.add_state(trend_model_);

    NEW(ZeroMeanGaussianConjSampler, observation_sampler)(
        model.observation_model(), observation_precision_prior_);
    model.observation_model()->set_method(observation_sampler);

    NEW(StateSpacePosteriorSampler, sampler)(&model);
    model.set_method(sampler);

    int niter = 200;
    Matrix trend_draws(niter, sample_size_);
    Vector sigma_obs_draws(niter);
    Vector sigma_level_draws(niter);
    Vector sigma_slope_draws(niter);
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      trend_draws.row(i) = model.state().row(0);
      sigma_obs_draws[i] = model.observation_model()->sigma();
      sigma_level_draws[i] = sqrt(trend_model_->Sigma()(0, 0));
      sigma_slope_draws[i] = sqrt(trend_model_->Sigma()(1, 1));
    }

    EXPECT_TRUE(CheckMcmcVector(sigma_level_draws, true_level_sd_));
    EXPECT_TRUE(CheckMcmcVector(sigma_slope_draws, true_slope_sd_));
    EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws, true_observation_sd_));
    auto status = CheckMcmcMatrix(trend_draws, trend_);
    EXPECT_TRUE(status.ok) << status;
  }
  //======================================================================
  TEST_F(LocalLinearTrendStateModelTest, DynamicInterceptRegressionModelTest) {
    SimulateData();
    BuildModel();

    Vector true_beta_ = {3, -2.4, 1.7};
    true_beta_.randomize();
    DynamicInterceptRegressionModel model(xdim_);
    NEW(LocalLinearTrendDynamicInterceptStateModel, trend_model)();
    trend_model->set_initial_state_mean(
        Vector{trend_[0], trend_[1] - trend_[0]});
    trend_model->set_initial_state_variance(
        SpdMatrix(2, var(trend_)));
        
    NEW(ZeroMeanMvnIndependenceSampler, level_sampler)(
          trend_model.get(), level_precision_prior_, 0);
    trend_model->set_method(level_sampler);
    NEW(ZeroMeanMvnIndependenceSampler, slope_sampler)(
        trend_model.get(), slope_precision_prior_, 1);
    trend_model->set_method(slope_sampler);

    model.add_state(trend_model);
    EXPECT_EQ(2, model.number_of_state_models());

    // Simulate the data.
    SpdMatrix xtx(xdim_, 0.0);
    double total_nobs = 0;
    for (int t = 0; t < sample_size_; ++t) {
      int nobs = 1 + rpois(3.0);
      Matrix predictors(nobs, xdim_);
      predictors.randomize();
      Vector response = predictors * true_beta_;
      response += trend_[t];
      for (int j = 0; j < nobs; ++j) {
        response[j] += rnorm_mt(GlobalRng::rng, 0, true_observation_sd_);
      }
      NEW(StateSpace::TimeSeriesRegressionData, data_point)(response, predictors);
      
      model.add_data(data_point);
      xtx += predictors.inner();
      total_nobs += nobs;
    }

    NEW(MvnGivenScalarSigma, coefficient_prior)(
        .01 * xtx / total_nobs, model.observation_model()->Sigsq_prm());
    NEW(RegressionConjSampler, regression_sampler)(
        model.observation_model(),
        coefficient_prior,
        observation_precision_prior_);
    model.observation_model()->set_method(regression_sampler);

    NEW(StateSpacePosteriorSampler, sampler)(&model);
    model.set_method(sampler);

    int niter = 500;
    Matrix trend_draws(niter, sample_size_);
    Vector sigma_obs_draws(niter);
    Vector sigma_level_draws(niter);
    Vector sigma_slope_draws(niter);
    Matrix beta_draws(niter, xdim_);
    Vector loglike_draws(niter);
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      loglike_draws[i] = model.log_likelihood();
      trend_draws.row(i) = model.state().row(1);
      sigma_obs_draws[i] = model.observation_model()->sigma();
      sigma_level_draws[i] = sqrt(trend_model->Sigma()(0, 0));
      sigma_slope_draws[i] = sqrt(trend_model->Sigma()(1, 1));
      beta_draws.row(i) = model.observation_model()->Beta();
    }
    std::ofstream loglike_file("loglike.txt");
    loglike_file << loglike_draws;
    
    EXPECT_TRUE(CheckMcmcVector(sigma_level_draws, true_level_sd_));
    EXPECT_GT(var(sigma_level_draws), 0.0);
    EXPECT_EQ("", CheckWithinRage(sigma_level_draws, .01, 10));
    
    EXPECT_TRUE(CheckMcmcVector(sigma_slope_draws, true_slope_sd_));
    EXPECT_GT(var(sigma_slope_draws), 0.0);
    EXPECT_EQ("", CheckWithinRage(sigma_slope_draws, .01, 10));
    
    EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws, true_observation_sd_,
                                .95, "sigma-obs.txt"))
        << AsciiDistributionCompare(sigma_obs_draws, true_observation_sd_);
    EXPECT_GT(var(sigma_obs_draws), 0.0);
    EXPECT_EQ("", CheckWithinRage(sigma_obs_draws, 0, 10));
              
    auto status = CheckMcmcMatrix(trend_draws, trend_, .95, true, "level.txt");
    EXPECT_TRUE(status.ok) << "LinearTrend did not cover. " << status;
    status = CheckMcmcMatrix(beta_draws, true_beta_, .95, true, "beta.txt");
    EXPECT_TRUE(status.ok) << "Coefficients did not cover" << status;
  }
}  // namespace

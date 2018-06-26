#include "gtest/gtest.h"

#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/DynamicInterceptRegression.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
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
  
  class LocalLevelStateModelTest : public ::testing::Test {
   protected:
    LocalLevelStateModelTest()
        : sample_size_(100),
          xdim_(3),
          true_level_sd_(.3),
          true_observation_sd_(1.4)
    {
      GlobalRng::rng.seed(8675309);
    }
    int sample_size_;
    int xdim_;
    double true_level_sd_;
    double true_observation_sd_;
    Ptr<ChisqModel> level_precision_prior_;
    Ptr<ChisqModel> observation_precision_prior_;
    Vector level_;
    Vector data_;
    Ptr<LocalLevelStateModel> level_model_;

    void SimulateData() {
      level_ = SimulateLocalLevel(GlobalRng::rng, sample_size_, 1.3,
                                  true_level_sd_);
      data_ = level_;
      for (int i = 0; i < data_.size(); ++i) {
        data_[i] += rnorm_mt(GlobalRng::rng, 0, true_observation_sd_);
      }
    }

    void BuildModel() {
      level_model_.reset(new LocalLevelStateModel(true_level_sd_));
      level_precision_prior_.reset(new ChisqModel(1.0, true_level_sd_));
      NEW(ZeroMeanGaussianConjSampler, level_sampler)(
          level_model_.get(), level_precision_prior_);
      level_model_->set_method(level_sampler);
      level_model_->set_initial_state_mean(1.3);
      level_model_->set_initial_state_variance(1.0);
          
      observation_precision_prior_.reset(new ChisqModel(
          1.0, true_observation_sd_));
    }

  };

  //======================================================================
  TEST_F(LocalLevelStateModelTest, ModelMatrices) {
    BuildModel();
    Matrix Id(1, 1, 1.0);
    level_model_->set_sigsq(4.0);
    EXPECT_TRUE(MatrixEquals(
        level_model_->state_transition_matrix(0)->dense(),
        Id));
    EXPECT_TRUE(MatrixEquals(
        level_model_->state_error_expander(0)->dense(),
        Id));

    SpdMatrix V(1, 4.0);
    EXPECT_TRUE(MatrixEquals(
        level_model_->state_variance_matrix(0)->dense(),
        V));
    EXPECT_TRUE(MatrixEquals(
        level_model_->state_error_variance(0)->dense(),
        V));
    EXPECT_EQ(1, level_model_->observation_matrix(0).size());
    EXPECT_DOUBLE_EQ(1.0, level_model_->observation_matrix(0)[0]);
  }

  //======================================================================
  TEST_F(LocalLevelStateModelTest, StateSpaceModelTest) {
    BuildModel();
    SimulateData();

    StateSpaceModel model(data_);
    model.add_state(level_model_);

    NEW(ZeroMeanGaussianConjSampler, observation_sampler)(
        model.observation_model(), observation_precision_prior_);
    model.observation_model()->set_method(observation_sampler);

    NEW(StateSpacePosteriorSampler, sampler)(&model);
    model.set_method(sampler);

    int niter = 200;
    Matrix level_draws(niter, sample_size_);
    Vector sigma_obs_draws(niter);
    Vector sigma_trend_draws(niter);
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      level_draws.row(i) = model.state().row(0);
      sigma_obs_draws[i] = model.observation_model()->sigma();
      sigma_trend_draws[i] = level_model_->sigma();
    }

    EXPECT_TRUE(CheckMcmcVector(sigma_trend_draws, true_level_sd_));
    EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws, true_observation_sd_));
    auto status = CheckMcmcMatrix(level_draws, level_);
    EXPECT_TRUE(status.ok) << status;
  }
  //======================================================================
  TEST_F(LocalLevelStateModelTest, DynamicInterceptRegressionModelTest) {
    BuildModel();
    SimulateData();

    Vector true_beta_ = {3, -2.4, 1.7};
    true_beta_.randomize();
    DynamicInterceptRegressionModel model(xdim_);
    NEW(LocalLevelDynamicInterceptStateModel, level_model)(1.0);
    level_model->set_initial_state_mean(level_[0]);
    level_model->set_initial_state_variance(var(level_));
    NEW(ZeroMeanGaussianConjSampler, level_sampler)(
        level_model.get(), level_precision_prior_);
    level_model->set_method(level_sampler);
    model.add_state(level_model);
    EXPECT_EQ(2, model.number_of_state_models());

    // Simulate the data.
    SpdMatrix xtx(xdim_, 0.0);
    double total_nobs = 0;
    for (int t = 0; t < sample_size_; ++t) {
      int nobs = 1 + rpois(2.0);
      Matrix predictors(nobs, xdim_);
      predictors.randomize();
      Vector response = predictors * true_beta_;
      response += level_[t];
      for (int j = 0; j < nobs; ++j) {
        response[j] += rnorm_mt(GlobalRng::rng, 0, true_observation_sd_);
      }
      NEW(StateSpace::TimeSeriesRegressionData, data_point)(response, predictors);
      
      model.add_data(data_point);
      xtx += predictors.inner();
      total_nobs += nobs;
    }

    cout << "total_nobs = " << total_nobs << endl;
    
    NEW(MvnGivenScalarSigma, coefficient_prior)(
        .01 * xtx / total_nobs, model.observation_model()->Sigsq_prm());
    NEW(RegressionConjSampler, regression_sampler)(
        model.observation_model(),
        coefficient_prior,
        observation_precision_prior_);
    model.observation_model()->set_method(regression_sampler);

    NEW(StateSpacePosteriorSampler, sampler)(&model);
    model.set_method(sampler);

    int niter = 200;
    Matrix level_draws(niter, sample_size_);
    Vector sigma_obs_draws(niter);
    Vector sigma_trend_draws(niter);
    Matrix beta_draws(niter, xdim_);
    Vector loglike_draws(niter);
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      loglike_draws[i] = model.log_likelihood();
      level_draws.row(i) = model.state().row(1);
      sigma_obs_draws[i] = model.observation_model()->sigma();
      sigma_trend_draws[i] = level_model->sigma();
      beta_draws.row(i) = model.observation_model()->Beta();
    }
    std::ofstream loglike_file("loglike.txt");
    loglike_file << loglike_draws;
    
    EXPECT_TRUE(CheckMcmcVector(sigma_trend_draws, true_level_sd_));
    EXPECT_GT(var(sigma_trend_draws), 0.0);
    EXPECT_EQ("", CheckWithinRage(sigma_trend_draws, .01, 10));
    
    EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws, true_observation_sd_,
                                .95, "sigma-obs.txt"))
        << AsciiDistributionCompare(sigma_obs_draws, true_observation_sd_);
    EXPECT_GT(var(sigma_obs_draws), 0.0);
    EXPECT_EQ("", CheckWithinRage(sigma_obs_draws, 0, 10));
              
    auto status = CheckMcmcMatrix(level_draws, level_, .95, true, "level.txt");
    EXPECT_TRUE(status.ok) << "Level did not cover. " << status;
    status = CheckMcmcMatrix(beta_draws, true_beta_, .95, true, "beta.txt");
    EXPECT_TRUE(status.ok) << "Coefficients did not cover" << status;
  }
}  // namespace

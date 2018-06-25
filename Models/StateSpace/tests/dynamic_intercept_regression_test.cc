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
#include "stats/AsciiDistributionCompare.hpp"

#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class DynamicInterceptRegressionModelTest : public ::testing::Test {
   protected:
    DynamicInterceptRegressionModelTest()
        : time_dimension_(20),
          true_level_sd_(.3),
          true_observation_sd_(1.4),
          true_beta_{3, -2.4, 1.7},
          level_precision_prior_(new ChisqModel(1.0, true_level_sd_)),
          residual_precision_prior_(new ChisqModel(1.0, true_observation_sd_)),
          spike_(new VariableSelectionPrior(Vector{.999, .999, .999}))
    {
      GlobalRng::rng.seed(8675309);
    }

    void SimulateData() {
      data_.clear();
      level_ = SimulateLocalLevel(
          GlobalRng::rng, time_dimension_, 1.3, true_level_sd_);
      for (int i = 0; i < time_dimension_; ++i) {
        int nobs = 1 + rpois(2.0);
        Matrix predictors(nobs, true_beta_.size());
        predictors.randomize();
        Vector response = predictors * true_beta_;
        response += level_[i];       
        for (int j = 0; j < response.size(); ++j) {
          response[j] += rnorm(0, true_observation_sd_);
        }
        NEW(StateSpace::TimeSeriesRegressionData, data_point)(
            response, predictors);
        data_.push_back(data_point);
      }
    }

    void BuildModel() {
      SimulateData();
      model_.reset(new DynamicInterceptRegressionModel(true_beta_.size()));
      for (const auto &data_point : data_) {
        model_->add_data(data_point);
      }
      
      level_model_.reset(new LocalLevelDynamicInterceptStateModel(
          true_level_sd_));
      level_model_->set_initial_state_mean(level_[0]);
      level_model_->set_initial_state_variance(var(level_));
      NEW(ZeroMeanGaussianConjSampler, level_sampler)(
          level_model_.get(), level_precision_prior_);
      level_sampler->set_sigma_upper_limit(10);
      level_model_->set_method(level_sampler);
      model_->add_state(level_model_);

      NEW(MvnGivenScalarSigma, coefficient_prior)(
          SpdMatrix(true_beta_.size(), .01),
          model_->observation_model()->Sigsq_prm());
      NEW(BregVsSampler, regression_sampler)(
          model_->observation_model(),
          coefficient_prior,
          residual_precision_prior_,
          spike_);
      regression_sampler->set_sigma_upper_limit(10);
      model_->observation_model()->set_method(regression_sampler);
      model_->observation_model()->set_Beta(true_beta_);
      model_->observation_model()->set_sigsq(square(true_observation_sd_));

      NEW(StateSpacePosteriorSampler, sampler)(model_.get());
      model_->set_method(sampler);
    }
      
    int time_dimension_;
    double true_level_sd_;
    double true_observation_sd_;
    Vector true_beta_;
    Vector level_;
    std::vector<Ptr<StateSpace::TimeSeriesRegressionData>> data_;
    Ptr<DynamicInterceptRegressionModel> model_;
    
    Ptr<LocalLevelDynamicInterceptStateModel> level_model_;
    Ptr<ChisqModel> level_precision_prior_;
    Ptr<ChisqModel> residual_precision_prior_;
    Ptr<VariableSelectionPrior> spike_;
  };

  TEST_F(DynamicInterceptRegressionModelTest, Coefficients) {
    NEW(DynamicInterceptRegressionModel, model)(7);
    EXPECT_EQ(7, model->observation_model()->Beta().size());

    BuildModel();
    EXPECT_EQ(model_->observation_model()->Beta().size(),
              true_beta_.size());
  }
  
  // With model parameters fixed at true values, check that the state can be
  // recovered.
  TEST_F(DynamicInterceptRegressionModelTest, DrawStateGivenParams) {
    BuildModel();
    level_model_->clear_methods();
    level_model_->set_sigsq(square(true_level_sd_));
    model_->observation_model()->clear_methods();
    model_->observation_model()->set_Beta(true_beta_);
    model_->observation_model()->set_sigsq(square(true_observation_sd_));

    // Check the observation coefficients.
    EXPECT_TRUE(VectorEquals(
        model_->dat()[0]->predictors() * true_beta_,
        model_->observation_coefficients(0)->dense().col(0)));
    
    int niter = 200;
    Matrix level_draws(niter, time_dimension_);
    for (int i = 0; i < niter; ++i) {
      model_->sample_posterior();
      level_draws.row(i) = model_->state().row(1);
    }
    auto status = CheckMcmcMatrix(level_draws, level_, .95);
    EXPECT_TRUE(status.ok) << "Level state component did not cover." << endl
                           << status;
    // Make sure the distribution isn't insanely wide.
    EXPECT_EQ("", CheckWithinRage(level_draws, level_ - 5, level_ + 5))
        << "True level = " << level_;
  }

  // With the state fixed at its true value, check that model parameters can be
  // recovered.
  TEST_F(DynamicInterceptRegressionModelTest, DrawParamsGivenState) {
    time_dimension_ = 40;
    BuildModel();
    Matrix true_state = rbind(level_, level_);
    true_state.row(0) = 1.0;
    model_->permanently_set_state(true_state);

    // Check the observation coefficients.
    EXPECT_TRUE(VectorEquals(
        model_->dat()[0]->predictors() * true_beta_,
        model_->observation_coefficients(0)->dense().col(0)));

    int niter = 200;
    Vector sigma_level_draws(niter);
    Vector sigma_obs_draws(niter);
    Matrix beta_draws(niter, true_beta_.size());
    for (int i = 0; i < niter; ++i) {
      model_->sample_posterior();
      sigma_level_draws[i] = level_model_->sigma();
      sigma_obs_draws[i] = model_->observation_model()->sigma();
      beta_draws.row(i) = model_->observation_model()->Beta();
    }

    EXPECT_TRUE(CheckMcmcVector(sigma_level_draws, true_level_sd_,
                                .95, "sigma-level.txt"));
    EXPECT_GT(var(sigma_level_draws), 0);
    EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws, true_observation_sd_,
                                .95, "sigma-obs.txt"));
    EXPECT_TRUE(var(sigma_obs_draws) > 0);
    auto status = CheckMcmcMatrix(beta_draws, true_beta_, .95, true, "beta.txt");
    EXPECT_TRUE(status.ok) << "Beta draws did not cover" << endl << status;
  }

  // With regression coefficients and observation fixed at their true values,
  // check that the state and the the level variance parameter are recovered.
  TEST_F(DynamicInterceptRegressionModelTest, FixedRegression) {
    SimulateData();
    BuildModel();
    model_->observation_model()->clear_methods();
    int niter = 200;
    Vector sigma_level_draws(niter);
    Matrix level_draws(niter, model_->time_dimension());
    for (int i = 0; i < niter; ++i) {
      model_->sample_posterior();
      sigma_level_draws[i] = level_model_->sigma();
      level_draws.row(i) = model_->state().row(1);
    }
    EXPECT_TRUE(VectorEquals(true_beta_, model_->observation_model()->Beta()));
    EXPECT_DOUBLE_EQ(true_observation_sd_, model_->observation_model()->sigma());
    EXPECT_TRUE(CheckMcmcVector(sigma_level_draws, true_level_sd_))
        << true_level_sd_ << " " << sigma_level_draws;
    auto status = CheckMcmcMatrix(level_draws, level_, .95, true,
                                  "state-with-fixed-regression.txt");
    EXPECT_TRUE(status.ok) << "State with fixed regression: " << status;
  }

  // A full MCMC check.
  TEST_F(DynamicInterceptRegressionModelTest, Mcmc) {
    SimulateData();
    BuildModel();
    EXPECT_EQ(model_->observation_model()->Beta().size(),
              true_beta_.size());

    int niter = 200;
    Vector sigma_level_draws(niter);
    Vector sigma_obs_draws(niter);
    Matrix beta_draws(niter, true_beta_.size());
    Matrix level_draws(niter, model_->time_dimension());
    for (int i = 0; i < niter; ++i) {
      model_->sample_posterior();
      sigma_level_draws[i] = level_model_->sigma();
      sigma_obs_draws[i] = model_->observation_model()->sigma();
      beta_draws.row(i) = model_->observation_model()->Beta();
      level_draws.row(i) = model_->state().row(1);
    }

    EXPECT_TRUE(CheckMcmcVector(sigma_level_draws, true_level_sd_,
                                .95, "sigma-level-mcmc.txt"))
        << endl << AsciiDistributionCompare(sigma_level_draws, true_level_sd_);
    EXPECT_GT(var(sigma_level_draws), 0);
    
    EXPECT_TRUE(CheckMcmcVector( sigma_obs_draws, true_observation_sd_, .95,
                                 "sigma-obs.txt"))
        << endl << AsciiDistributionCompare(
            sigma_obs_draws, true_observation_sd_);
    EXPECT_TRUE(var(sigma_obs_draws) > 0);

    auto status = CheckMcmcMatrix(beta_draws, true_beta_, .95, true,
                                  "beta-mcmc.txt");
    EXPECT_TRUE(status.ok) << "Beta draws did not cover" << endl << status;

    status  = CheckMcmcMatrix(level_draws, level_);
    EXPECT_TRUE(status.ok) << "State did not cover" << endl << status;
  }
  
}  // namespace

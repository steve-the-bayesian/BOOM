#include "gtest/gtest.h"

#include "Models/ChisqModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "Models/StateSpace/StateModels/DynamicRegressionStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionPosteriorSampler.hpp"

#include "cpputil/Constants.hpp"
#include "cpputil/Date.hpp"
#include "cpputil/seq.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include "Models/StateSpace/tests/state_space_test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class DynamicRegressionStateModelTest : public ::testing::Test {
   protected:
    DynamicRegressionStateModelTest()
        : sample_size_(100),
          xdim_(4),
          true_trend_sd_(.3),
          true_observation_sd_(.2),
          true_coefficient_sd_(.8)
    {
      GlobalRng::rng.seed(8675309);
    }

    int sample_size_;
    int xdim_;
    double true_trend_sd_;
    double true_observation_sd_;
    double true_coefficient_sd_;
    Vector trend_;
    Matrix coefficients_;
    Matrix predictors_;
    Vector regression_;
    Vector data_;
    Ptr<LocalLevelStateModel> level_model_;
    Ptr<DynamicRegressionStateModel> dynamic_regression_model_;
    Ptr<StateSpaceModel> model_;

    void SimulateData() {
      trend_ = SimulateLocalLevel(GlobalRng::rng, sample_size_, 0.0,
                                  true_trend_sd_);
      predictors_.resize(sample_size_, xdim_);
      predictors_.randomize();

      coefficients_.resize(xdim_, sample_size_);
      for (int i = 0; i < xdim_; ++i) {
        coefficients_.row(i) = SimulateLocalLevel(
            GlobalRng::rng, sample_size_, 0.0, true_coefficient_sd_);
      }

      regression_.resize(sample_size_);
      data_.resize(sample_size_);
      for (int i = 0; i < sample_size_; ++i) {
        regression_[i] = coefficients_.col(i).dot(predictors_.row(i));
        data_[i] = trend_[i] + regression_[i]
            + rnorm_mt(GlobalRng::rng, 0, true_observation_sd_);
      }
    }

    void BuildModel() {
      model_.reset(new StateSpaceModel(data_));
      level_model_.reset(new LocalLevelStateModel);
      NEW(ChisqModel, level_precision_prior)(1.0, true_trend_sd_);
      NEW(ZeroMeanGaussianConjSampler, level_sampler)(
          level_model_.get(), level_precision_prior);
      level_model_->set_method(level_sampler);
      model_->add_state(level_model_);

      dynamic_regression_model_.reset(new DynamicRegressionStateModel(
          predictors_));
      NEW(ChisqModel, dynamic_regression_precision_prior_)(
          1.0, true_coefficient_sd_);
      NEW(DynamicRegressionIndependentPosteriorSampler,
          dynamic_regression_samper)(
              dynamic_regression_model_.get(),
              std::vector<Ptr<GammaModelBase>>(
                  xdim_, dynamic_regression_precision_prior_));
      dynamic_regression_model_->set_method(dynamic_regression_samper);
      model_->add_state(dynamic_regression_model_);

      NEW(ChisqModel, observation_precision_prior)(1.0, true_observation_sd_);
      NEW(ZeroMeanGaussianConjSampler, observation_precision_sampler)(
          model_->observation_model(), observation_precision_prior);
      model_->observation_model()->set_method(observation_precision_sampler);

      NEW(StateSpacePosteriorSampler, sampler)(model_.get());
      model_->set_method(sampler);
    }

  };

  //======================================================================
  TEST_F(DynamicRegressionStateModelTest, FullMcmc) {
    SimulateData();
    BuildModel();

    int niter = 400;
    Vector sigma_obs_draws(niter);
    Vector sigma_level_draws(niter);
    Matrix coefficient_sd_draws(niter, xdim_);
    Matrix trend_draws(niter, sample_size_);
    Matrix coefficient_draws_1(niter, sample_size_);
    Matrix coefficient_draws_2(niter, sample_size_);
    Matrix coefficient_draws_3(niter, sample_size_);
    Matrix coefficient_draws_4(niter, sample_size_);
    Matrix regression_draws(niter, sample_size_);
    
    for (int i = 0; i < 100; ++i) {
      model_->sample_posterior();
    }
    for (int i = 0; i < niter; ++i) {
      model_->sample_posterior();
      sigma_obs_draws[i] = model_->observation_model()->sigma();
      sigma_level_draws[i] = level_model_->sigma();
      for (int j = 0; j < xdim_; ++j) {
        coefficient_sd_draws(i, j) =
            std::sqrt(dynamic_regression_model_->sigsq(j));
      }
      trend_draws.row(i) = model_->state().row(0);
      coefficient_draws_1.row(i) = model_->state().row(1);
      coefficient_draws_2.row(i) = model_->state().row(2);
      coefficient_draws_3.row(i) = model_->state().row(3);
      coefficient_draws_4.row(i) = model_->state().row(4);
      for (int t = 0; t < sample_size_; ++t) {
        ConstVectorView coefficients(model_->state(t), 1);
        regression_draws(i, t) = predictors_.row(t).dot(coefficients);
      }
    }

    EXPECT_TRUE(CheckMcmcVector(sigma_obs_draws, true_observation_sd_,
                                .98, "sigma_obs_draws.txt"));
    EXPECT_TRUE(CheckMcmcVector(sigma_level_draws, true_trend_sd_));
    auto coef_sd_status = CheckMcmcMatrix(
        coefficient_sd_draws, Vector(xdim_, true_coefficient_sd_));
    EXPECT_TRUE(coef_sd_status.ok) << coef_sd_status;

    auto trend_status = CheckMcmcMatrix(trend_draws, trend_);
    EXPECT_TRUE(trend_status.ok) << trend_status;

    auto regression_status = CheckMcmcMatrix(regression_draws, regression_);
    EXPECT_TRUE(regression_status.ok) << regression_status;
    
    // Check for wildness
    double sample_var = var(data_);
    for (int t = 0; t < sample_size_; ++t) {
      EXPECT_LT(var(regression_draws.col(t)), sample_var);
    }
  }
}  // namespace

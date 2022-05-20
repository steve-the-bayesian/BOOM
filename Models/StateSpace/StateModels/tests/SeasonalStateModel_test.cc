#include "gtest/gtest.h"

#include <fstream>

#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "stats/moments.hpp"
#include "stats/AsciiDistributionCompare.hpp"

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/LocalLevelModule.hpp"
#include "Models/StateSpace/StateModels/test_utils/StaticInterceptTestModule.hpp"
#include "Models/StateSpace/StateModels/test_utils/SeasonalTestModule.hpp"

#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;

  class SeasonalTest : public ::testing::Test {
   protected:
    SeasonalTest()
        : weeks_per_year_(18),
          day_of_week_cycle_(new SeasonalStateModel(7, 1)),
          weekly_annual_cycle_(new SeasonalStateModel(weeks_per_year_, 7)),
          sigma_obs_(.5)
    {
      GlobalRng::rng.seed(8675309);
      day_of_week_cycle_->set_sigsq(square(.1));
      EXPECT_EQ(6, day_of_week_cycle_->state_dimension());
      weekly_annual_cycle_->set_sigsq(square(.25));
      SetInitialDistribution();
      SimulateData();
    }

    void SetInitialDistribution() {
      day_of_week_cycle_->set_initial_state_mean(
          Vector(day_of_week_cycle_->state_dimension(), 0.0));
      day_of_week_cycle_->set_initial_state_variance(
          SpdMatrix(day_of_week_cycle_->state_dimension(), square(2.0)));
      weekly_annual_cycle_->set_initial_state_mean(
          Vector(weekly_annual_cycle_->state_dimension(), 0.0));
      weekly_annual_cycle_->set_initial_state_variance(
          SpdMatrix(weekly_annual_cycle_->state_dimension(), square(2.0)));
    }

    void SimulateData() {
      nobs_ = 500;
      int state_dimension = day_of_week_cycle_->state_dimension() +
          weekly_annual_cycle_->state_dimension();
      true_state_ = Matrix(state_dimension, nobs_);
      series_ = Vector(nobs_);

      Vector state(state_dimension);
      day_of_week_cycle_->simulate_initial_state(
          GlobalRng::rng, VectorView(state, 0, day_of_week_cycle_->state_dimension()));
      weekly_annual_cycle_->simulate_initial_state(
          GlobalRng::rng, VectorView(state, day_of_week_cycle_->state_dimension(),
                                     weekly_annual_cycle_->state_dimension()));
      true_state_.col(0) = state;

      for (int i = 0; i < nobs_; ++i) {
        true_state_.col(i) = state;
        series_[i] = state[0] + state[6] + rnorm(0, sigma_obs_);

        if (i < nobs_ - 1) {
          BlockDiagonalMatrix transition;
          transition.add_block(day_of_week_cycle_->state_transition_matrix(i));
          transition.add_block(weekly_annual_cycle_->state_transition_matrix(i));
          state = transition * state;

          Vector state_error(state_dimension);
          day_of_week_cycle_->simulate_state_error(
              GlobalRng::rng,
              VectorView(state_error, 0, day_of_week_cycle_->state_dimension()),
              i);
          weekly_annual_cycle_->simulate_state_error(
              GlobalRng::rng,
              VectorView(state_error, day_of_week_cycle_->state_dimension(),
                         weekly_annual_cycle_->state_dimension()),
              i);
          state += state_error;
        }
      }
    }

    // Make weeks_per_year_ smaller than 52 to keep the test running fast.
    int weeks_per_year_;
    Ptr<SeasonalStateModel> day_of_week_cycle_;
    Ptr<SeasonalStateModel> weekly_annual_cycle_;
    double sigma_obs_;

    int nobs_;
    Matrix true_state_;
    Vector series_;
  };

  TEST_F(SeasonalTest, Basics) {
    EXPECT_EQ(7 - 1, day_of_week_cycle_->state_dimension());
    EXPECT_EQ(1, day_of_week_cycle_->state_error_dimension());
    EXPECT_EQ(weeks_per_year_ - 1, weekly_annual_cycle_->state_dimension());
    EXPECT_EQ(1, weekly_annual_cycle_->state_error_dimension());

    EXPECT_TRUE(day_of_week_cycle_->new_season(0));
    EXPECT_TRUE(day_of_week_cycle_->new_season(1));
    EXPECT_TRUE(day_of_week_cycle_->new_season(2));
    EXPECT_TRUE(day_of_week_cycle_->new_season(3));

    EXPECT_TRUE(weekly_annual_cycle_->new_season(0));
    EXPECT_FALSE(weekly_annual_cycle_->new_season(1));
    EXPECT_FALSE(weekly_annual_cycle_->new_season(2));
    EXPECT_FALSE(weekly_annual_cycle_->new_season(3));
    EXPECT_FALSE(weekly_annual_cycle_->new_season(4));
    EXPECT_FALSE(weekly_annual_cycle_->new_season(5));
    EXPECT_FALSE(weekly_annual_cycle_->new_season(6));
    EXPECT_TRUE(weekly_annual_cycle_->new_season(7));
    EXPECT_FALSE(weekly_annual_cycle_->new_season(8));
    EXPECT_TRUE(weekly_annual_cycle_->new_season(14));

    Vector state_error(weekly_annual_cycle_->state_dimension());
    weekly_annual_cycle_->simulate_state_error(
        GlobalRng::rng, VectorView(state_error), 0);
    EXPECT_DOUBLE_EQ(state_error[0], 0.0);

    weekly_annual_cycle_->simulate_state_error(
        GlobalRng::rng, VectorView(state_error), 1);
    EXPECT_DOUBLE_EQ(state_error[0], 0.0);

    weekly_annual_cycle_->simulate_state_error(
        GlobalRng::rng, VectorView(state_error), 6);
    EXPECT_NE(state_error[0], 0.0);

    weekly_annual_cycle_->simulate_state_error(
        GlobalRng::rng, VectorView(state_error), 7);
    EXPECT_DOUBLE_EQ(state_error[0], 0.0);
  }

  //===========================================================================
  // Simulate data from a 4-cycle with a 3-day duration.  Check that the cycle
  // is correctly recovered, and that predictions from the model are reasonable.
  TEST_F(SeasonalTest, SeasonDuration) {
    //-----------  Simulate the data ------------------
    Vector seasonal_frame(103);
    seasonal_frame[0] = -100;
    seasonal_frame[1] = 20;
    seasonal_frame[2] = 50;

    double true_seasonal_sd = .25;
    for (int i = 3; i < seasonal_frame.size(); ++i) {
      double past = seasonal_frame[i - 1] + seasonal_frame[i - 2]
          + seasonal_frame[i - 3];
      seasonal_frame[i] = rnorm(-past, true_seasonal_sd);
    }
    Vector seasonal(300);
    int frame_cursor = 3;
    for (int i = 0; i < 300; ++i) {
      if (i % 3 == 0) {
        seasonal[i] = seasonal_frame[frame_cursor++];
      } else {
        seasonal[i] = seasonal[i - 1];
      }
    }

    double true_level_sd = .1;
    Vector trend(300);
    trend[0] = 0;
    for (int i = 1; i < 300; ++i) {
      trend[i] = trend[i - 1] + rnorm(0, true_level_sd);
    }

    double true_sigma_obs = .2;
    Vector data(300);
    for (int i = 0; i < 300; ++i) {
      data[i] = trend[i] + seasonal[i] + rnorm(0, true_sigma_obs);
    }

    Vector training_data(ConstVectorView(data, 0, 275));
    Vector holdout_data(ConstVectorView(data, 275, 25));

    //-----------  Build the model ------------------

    NEW(StateSpaceModel, model)(training_data);
    // Add a local level trend.
    NEW(LocalLevelStateModel, level)(1.0);
    NEW(ChisqModel, level_precision_prior)(1.0, true_level_sd);
    NEW(ZeroMeanGaussianConjSampler, level_sampler)(
        level.get(),
        level_precision_prior);
    level->set_method(level_sampler);
    level->set_initial_state_mean(mean(data));
    level->set_initial_state_variance(var(data));
    model->add_state(level);

    // Add seasonal effect.
    NEW(SeasonalStateModel, seasonal_model)(4, 3);
    NEW(ChisqModel, seasonal_precision_prior)(1, true_seasonal_sd);
    NEW(ZeroMeanGaussianConjSampler, seasonal_sampler)(
        seasonal_model.get(),
        seasonal_precision_prior);
    seasonal_model->set_method(seasonal_sampler);
    seasonal_model->set_initial_state_mean(Vector(3, 0.0));
    seasonal_model->set_initial_state_variance(SpdMatrix(3, var(data)));
    model->add_state(seasonal_model);

    // Sampler for observation model.
    NEW(ChisqModel, observation_precision_prior)(1, true_sigma_obs);
    NEW(ZeroMeanGaussianConjSampler, observation_model_sampler)(
        model->observation_model(),
        observation_precision_prior);
    model->observation_model()->set_method(observation_model_sampler);

    // Sampler for overall model.
    NEW(StateSpacePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    //-------------------- MCMC --------------------
    int burn = 100;
    for (int i = 0; i < burn; ++i) {
      model->sample_posterior();
    }

    int niter = 500;
    int time_dimension = training_data.size();
    int horizon = holdout_data.size();
    Matrix level_draws(niter, time_dimension);
    Matrix seasonal_draws(niter, time_dimension);
    Vector level_sd_draws(niter);
    Vector seasonal_sd_draws(niter);
    Matrix posterior_predictive(niter, horizon);

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      level_draws.row(i) = model->state().row(0);
      seasonal_draws.row(i) = model->state().row(1);
      level_sd_draws[i] = level->sigma();
      seasonal_sd_draws[i] = seasonal_model->sigma();
      posterior_predictive.row(i) = model->simulate_forecast(
          sampler->rng(), horizon, model->final_state());
    }

    //-------------------- Check the results --------------------
    CheckMatrixStatus level_status = CheckMcmcMatrix(
        level_draws, ConstVectorView(trend, 0, 275), .95, true,
        "level_draws.out");
    EXPECT_TRUE(level_status.ok)
        << "Level failed to cover: " << endl << level_status.error_message();

    std::string error_message = CheckStochasticProcess(
        seasonal_draws, ConstVectorView(seasonal, 0, 275),
        .95, .1, 0.5, "seasonal_draws.txt");
    EXPECT_EQ("", error_message) << "Seasonal pattern failed to cover.";

    if (!(level_status.ok && error_message == "")) {
      std::ofstream raw_data_file("raw_data.txt");
      raw_data_file << data;
      std::ofstream level_file("level.txt");
      level_file << trend << endl << level_draws;
      std::ofstream seasonal_file("seasonal.txt");
      seasonal_file << seasonal << endl << seasonal_draws;
    }

    bool level_sd_ok = CheckMcmcVector(level_sd_draws, .1);
    EXPECT_TRUE(level_sd_ok)
        << AsciiDistributionCompare(level_sd_draws, .1);
    bool seasonal_sd_ok = CheckMcmcVector(seasonal_sd_draws, .25);
    EXPECT_TRUE(seasonal_sd_ok)
        << AsciiDistributionCompare(seasonal_sd_draws, .25);

    if (!(level_sd_ok && seasonal_sd_ok)) {
      std::ofstream level_sd_file("level_sd.txt");
      level_sd_file << .1 << endl << level_sd_draws;
      std::ofstream seasonal_sd_file("seasonal_sd.txt");
      seasonal_sd_file << .25 << endl << seasonal_sd_draws;
    }

    //-------------------- Forecast ----------------------
    CheckMatrixStatus forecast_status = CheckMcmcMatrix(
        posterior_predictive, holdout_data);
    EXPECT_TRUE(forecast_status.ok)
        << "Coverage error for full forecast"
        << forecast_status.error_message();

    forecast_status = CheckMcmcMatrix(
        SubMatrix(posterior_predictive, 0, niter - 1, 0, 5).to_matrix(),
        Vector(ConstVectorView(holdout_data, 0, 6)));
    EXPECT_TRUE(forecast_status.ok)
        << "Coverage error for 6-step time horizon"
        << forecast_status.error_message();
  }

  //===========================================================================
  TEST_F(SeasonalTest, Framework) {
    double true_sigma_obs = 1.2;
    StateSpaceTestFramework state_space(true_sigma_obs);
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules_;
    modules_.AddModule(new StaticInterceptTestModule(3.7));
    modules_.AddModule(new SeasonalTestModule(.8, 7));
    modules_.AddModule(new SeasonalTestModule(1.1, weeks_per_year_, 7));
    state_space.AddState(modules_);
    int niter = 600;
    int time_dimension = 400;
    state_space.Test(niter, time_dimension);
  }

  //===========================================================================
  TEST_F(SeasonalTest, Mcmc) {
    NEW(StateSpaceModel, model)(series_);

    NEW(SeasonalStateModel, daily_model)(7);
    NEW(ChisqModel, daily_precision_prior)(1, 1);
    NEW(ZeroMeanGaussianConjSampler, daily_sampler)(
        daily_model.get(), daily_precision_prior);
    daily_model->set_method(daily_sampler);
    daily_model->set_initial_state_mean(
        day_of_week_cycle_->initial_state_mean());
    daily_model->set_initial_state_variance(
        day_of_week_cycle_->initial_state_variance());
    model->add_state(daily_model);

    NEW(SeasonalStateModel, weekly_model)(weeks_per_year_, 7);
    NEW(ChisqModel, weekly_precision_prior)(1, 1);
    NEW(ZeroMeanGaussianConjSampler, weekly_sampler)(
        weekly_model.get(), weekly_precision_prior);
    weekly_model->set_method(weekly_sampler);
    weekly_model->set_initial_state_mean(
        weekly_annual_cycle_->initial_state_mean());
    weekly_model->set_initial_state_variance(
        weekly_annual_cycle_->initial_state_variance());
    model->add_state(weekly_model);

    NEW(ChisqModel, observation_precision)(1, 1);
    NEW(ZeroMeanGaussianConjSampler, obs_sampler)(
        model->observation_model(),
        observation_precision);
    model->observation_model()->set_method(obs_sampler);

    NEW(StateSpacePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    model->sample_posterior();

    int niter = 500;
    Matrix day_of_week_draws(niter, series_.size());
    Matrix weekly_draws(niter, series_.size());
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      day_of_week_draws.row(i) = model->state().row(0);
      weekly_draws.row(i) = model->state().row(6);
    }

    bool both_ok = true;
    CheckMatrixStatus status = CheckMcmcMatrix(day_of_week_draws,
                                               true_state_.row(0));
    EXPECT_TRUE(status.ok)
        << "Day of week pattern failed to cover." << endl
        << status.error_message();
    both_ok &= status.ok;

    std::string error_message = CheckStochasticProcess(weekly_draws, true_state_.row(6));
    EXPECT_EQ(error_message, "")
        << "Weekly annual cycle failed to cover." << endl
        << status.error_message();
    both_ok &= error_message == "";

    if (!both_ok) {
      std::ofstream series_file("raw.data");
      series_file << series_;
      std::ofstream day_file("day.draws");
      day_file << true_state_.row(0) << endl << day_of_week_draws;
      std::ofstream week_file("week.draws");
      week_file << true_state_.row(6) << endl << weekly_draws;
    }
  }


}  // namespace

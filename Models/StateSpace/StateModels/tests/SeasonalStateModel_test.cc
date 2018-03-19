#include "gtest/gtest.h"
#include "Models/ChisqModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class SeasonalTest : public ::testing::Test {
   protected:
    SeasonalTest()
        : day_of_week_cycle_(new SeasonalStateModel(7, 1)),
          weekly_annual_cycle_(new SeasonalStateModel(52, 7)),
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
      nobs_ = 1000;
      int state_dimension = day_of_week_cycle_->state_dimension() +
          weekly_annual_cycle_->state_dimension();
      int state_error_dimension = day_of_week_cycle_->state_error_dimension() +
          weekly_annual_cycle_->state_error_dimension();
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
        series_[i] = state[0] + state[7] + rnorm(0, sigma_obs_);

        if (i < nobs_ - 1) {
          BlockDiagonalMatrix transition;
          transition.add_block(day_of_week_cycle_->state_transition_matrix(i));
          transition.add_block(weekly_annual_cycle_->state_transition_matrix(i));
          state = transition * state;
          
          Vector state_error(state_error_dimension);
          day_of_week_cycle_->simulate_state_error(
              GlobalRng::rng,
              VectorView(state_error, 0, day_of_week_cycle_->state_error_dimension()),
              i);
          weekly_annual_cycle_->simulate_state_error(
              GlobalRng::rng,
              VectorView(state_error, day_of_week_cycle_->state_error_dimension(),
                         weekly_annual_cycle_->state_error_dimension()),
              i);

          BlockDiagonalMatrix error_expander;
          error_expander.add_block(day_of_week_cycle_->state_error_expander(i));
          error_expander.add_block(weekly_annual_cycle_->state_error_expander(i));
          state += error_expander * state_error;
        }
      }
    }
    
    Ptr<SeasonalStateModel> day_of_week_cycle_;
    Ptr<SeasonalStateModel> weekly_annual_cycle_;
    double sigma_obs_;

    int nobs_;
    Vector daily_pattern_;  // Length 7
    Vector lunar_pattern_;  // length 12, season_duration = 7
    Matrix true_state_;
    Vector series_;
  };

  TEST_F(SeasonalTest, Basics) {
    EXPECT_EQ(7 - 1, day_of_week_cycle_->state_dimension());
    EXPECT_EQ(1, day_of_week_cycle_->state_error_dimension());
    EXPECT_EQ(52 - 1, weekly_annual_cycle_->state_dimension());
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

    Vector state_error(1);
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

    NEW(SeasonalStateModel, weekly_model)(52, 7);
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
      weekly_draws.row(i) = model->state().row(7);
    }

    std::ofstream day_file("day.draws");
    day_file << day_of_week_draws;
    std::ofstream week_file("week.draws");
    week_file << weekly_draws;
  }
  
}  // namespace

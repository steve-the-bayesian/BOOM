#include "gtest/gtest.h"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
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
      day_of_week_cycle_->set_initial_state_mean(
          Vector(day_of_week_cycle_->state_dimension(), 0.0));
      day_of_week_cycle_->set_initial_state_variance(
          SpdMatrix(day_of_week_cycle_->state_dimension(), square(2.0)));
          
      weekly_annual_cycle_->set_sigsq(square(.25));
      weekly_annual_cycle_->set_initial_state_mean(
          Vector(weekly_annual_cycle_->state_dimension(), 0.0));
      weekly_annual_cycle_->set_initial_state_variance(
          SpdMatrix(weekly_annual_cycle_->state_dimension(), square(2.0)));

      SimulateData();
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

  TEST_F(SeasonalTest, Basics) { }
  
}  // namespace

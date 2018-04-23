#include "gtest/gtest.h"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModels/LocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/HMM/GeneralHmmStateSpaceWrapper.hpp"
#include "Models/HMM/PosteriorSamplers/LiuWestParticleFilter.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class LiuWestParticleFilterTest : public ::testing::Test {
   protected:
    LiuWestParticleFilterTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(LiuWestParticleFilterTest, Basics) {
  }

  TEST_F(LiuWestParticleFilterTest, Filtering) {
    bool save_to_file = false;

    Vector true_state(100);
    Vector observations(true_state.size());
    std::vector<Ptr<DoubleData>> data;
    double true_sigma_obs = .5;
    double true_sigma_trend = .1;
    for (int i = 0; i < true_state.size(); ++i) {
      true_state[i] = (i == 0) ? rnorm(0, 3) : rnorm(true_state[i-1], true_sigma_trend);
      observations[i] = true_state[i] + rnorm(0, true_sigma_obs);
      data.push_back(new DoubleData(observations[i]));
    }

    NEW(StateSpaceModel, base_model)();
    EXPECT_TRUE(base_model->observation_model() != nullptr);
    EXPECT_DOUBLE_EQ(base_model->observation_model()->sigsq(), 1.0);
    
    NEW(LocalLevelStateModel, trend)();
    base_model->add_state(trend);
    NEW(GeneralHmmStateSpaceWrapper, model)(base_model);

    int number_of_particles = 1500;
    LiuWestParticleFilter filter(model, number_of_particles, .1);
    Matrix initial_parameters(number_of_particles, 2);
    Matrix initial_state(number_of_particles, 1);

    // The parameters of the model are
    // 1) a scalar observation variance.
    // 2) a scalar trend innovation variance.
    for (int i = 0; i < number_of_particles; ++i) {
      initial_parameters.row(i)[0] = 1.0 / rgamma(2, 1);
      initial_parameters.row(i)[1] = 1.0 / rgamma(2, .01);
      initial_state.row(i)[0] = rnorm(0, 10);
    }
    filter.set_particles(initial_state, initial_parameters);
    
    std::ofstream state_file;
    std::ofstream sigma_obs_file;
    std::ofstream sigma_trend_file;
    std::ofstream weight_file;
    if (save_to_file) {
      state_file.open("state.out");
      sigma_obs_file.open("sigma_obs.out");
      sigma_trend_file.open("sigma_trend.out");
      weight_file.open("weights.out");
    }
    
    Matrix state_draws(number_of_particles, true_state.size());
    for (int i = 0; i < data.size(); ++i) {
      filter.update(GlobalRng::rng, *data[i], i);
      Vector state_distribution = filter.state_distribution().col(0);
      state_draws.col(i) = state_distribution;
      if (save_to_file) {
        state_file << true_state[i] << " " << state_distribution << endl;
        Matrix parameter_distribution =
            filter.parameter_distribution(&GlobalRng::rng);
        sigma_obs_file << true_sigma_obs << ' '
                       << sqrt(parameter_distribution.col(0)) << endl;
        sigma_trend_file << true_sigma_trend << ' '
                         << sqrt(parameter_distribution.col(1)) << endl;
        weight_file << filter.particle_weights() << endl;
      }
    }

    if (save_to_file) {
      std::ofstream trend_file("trend.out");
      trend_file << true_state;

      std::ofstream data_file("data.out");
      data_file << observations;
    }
    Matrix parameter_distribution = filter.parameter_distribution();
    EXPECT_TRUE(CheckMcmcVector(sqrt(parameter_distribution.col(0)),
                                true_sigma_obs))
        << "Sigma obs didn't cover.";

    EXPECT_TRUE(CheckMcmcVector(sqrt(parameter_distribution.col(1)),
                                true_sigma_trend))
        << "Sigma trend didn't cover.";

    auto status = CheckMcmcMatrix(state_draws, true_state);
    EXPECT_TRUE(status.ok)
        << "State failed to cover" << endl
        << status;
  }

  /*
    PlotParameter <- function(fname, burn = 0) {
      ## An R function that can be used to plot the data written by this test.
      library(Boom)
      prm <- t(mscan(fname))
      if (burn > 0) {
        prm <- prm[, -(1:burn)]
      }
      truth <- prm[1, ]
      PlotDynamicDistribution(prm)
      lines(truth, col = "green", lwd = 3)
    }
   */

  
}  // namespace

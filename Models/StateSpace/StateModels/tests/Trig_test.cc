#include "gtest/gtest.h"

#include "Models/ChisqModel.hpp"
#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "Models/StateSpace/StateModels/TrigStateModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "cpputil/Constants.hpp"
#include "cpputil/Date.hpp"
#include "cpputil/seq.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

#include "Models/StateSpace/tests/StateSpaceTestFramework.hpp"
#include "Models/StateSpace/StateModels/test_utils/TrigTestModule.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpaceTesting;
  using std::endl;
  using std::cout;
  
  class TrigStateModelTest : public ::testing::Test {
   protected:
    TrigStateModelTest()
        : period_(365.25),
          frequencies_(seq<double>(1.0, 12.0)),
          coefficient_innovation_precision_prior_(new ChisqModel(10.0, 3.0))
    {
      GlobalRng::rng.seed(8675309);
    }

    double period_;
    Vector frequencies_;
    Ptr<ChisqModel> coefficient_innovation_precision_prior_;
    std::vector<Ptr<GammaModelBase>> specific_coefficient_precision_priors_;
    Ptr<IndependentMvnVarSampler> coefficient_precision_sampler_;
  };

  inline double dsquare(double x) {return x * x;}
  //======================================================================
  TEST_F(TrigStateModelTest, ModelMatrices) {
    TrigRegressionStateModel trig(period_, frequencies_);
    EXPECT_EQ(trig.state_dimension(), 2 * frequencies_.size());

    SparseVector Z = trig.observation_matrix(17.0);
    // Successive elements of Z are pairs of sines and cosines.
    EXPECT_EQ(Z.size(), 2 * frequencies_.size());
    EXPECT_DOUBLE_EQ(dsquare(Z[0]) + dsquare(Z[1]), 1.0);
    EXPECT_DOUBLE_EQ(dsquare(Z[2]) + dsquare(Z[3]), 1.0);
    EXPECT_DOUBLE_EQ(dsquare(Z[4]) + dsquare(Z[5]), 1.0);

    EXPECT_DOUBLE_EQ(0.0, trig.suf()->n());
    for (int i = 0; i < trig.state_dimension(); ++i) {
      EXPECT_DOUBLE_EQ(0.0, trig.suf()->sum(i));
      EXPECT_DOUBLE_EQ(0.0, trig.suf()->sumsq(i));
    }
    Vector then = seq<double>(1.0, trig.state_dimension());
    Vector now = then + 2;
    trig.observe_state(then, now, 3);
    EXPECT_DOUBLE_EQ(1.0, trig.suf()->n());
    for (int i = 0; i < trig.state_dimension(); ++i) {
      EXPECT_DOUBLE_EQ(2, trig.suf()->sum(i));
      EXPECT_DOUBLE_EQ(4, trig.suf()->sumsq(i));
    }
  }

  //======================================================================
  TEST_F(TrigStateModelTest, StateSpaceFramework) {
    StateSpaceTestFramework framework(1.2);
    StateModuleManager<StateModel, ScalarStateSpaceModelBase> modules;
    int time_dimension = 300;
    double period = time_dimension / 5.0;
    Vector frequencies = {1, 2};
    modules.AddModule(new TrigTestModule(period, frequencies, 0.3));
    framework.AddState(modules);
    int niter = 500;
    framework.Test(niter, time_dimension);
  }
  //======================================================================
  TEST_F(TrigStateModelTest, HarmonicTrigMCMC) {
    int time_dimension = 200;

    Vector first_harmonic(time_dimension);
    Vector second_harmonic(time_dimension);
    Vector y(time_dimension);
    int period = time_dimension / 5;
    double residual_sd = 1;
    
    for (int t = 0; t < time_dimension; ++t) {
      double freq1 = 2.0 * Constants::pi * t / period;
      double freq2 = 2 * freq1; 
      first_harmonic[t] = 3.2 * cos(freq1) - 1.6 * sin(freq1);
      second_harmonic[t] = -.8 * cos(freq2) + .25 * sin(freq2);
      y[t] = first_harmonic[t] + second_harmonic[t]
          + rnorm_mt(GlobalRng::rng, 0, residual_sd);
    }

    StateSpaceModel model(y);
    NEW(TrigStateModel, trig_state)(period, {1.0, 2.0});
    trig_state->set_initial_state_mean(
        Vector(trig_state->state_dimension(), 0.0));
    trig_state->set_initial_state_variance(
        SpdMatrix(trig_state->state_dimension(), 100.0));
    
    NEW(ChisqModel, innovation_precision_prior)(1, .1);
    NEW(ZeroMeanGaussianConjSampler, state_variance_sampler)(
        trig_state->error_distribution(),
        innovation_precision_prior);
    trig_state->set_method(state_variance_sampler);
    trig_state->error_distribution()->set_method(state_variance_sampler);
    model.add_state(trig_state);

    NEW(ChisqModel, observation_precision_prior)(1, 0.2);
    NEW(ZeroMeanGaussianConjSampler, sigma_obs_sampler)(
        model.observation_model(),
        observation_precision_prior);
    model.observation_model()->set_method(sigma_obs_sampler);
    
    NEW(StateSpacePosteriorSampler, sampler)(&model);
    model.set_method(sampler);

    EXPECT_EQ(model.state_dimension(), 4);
    
    int niter = 500;
    Matrix first_harmonic_state_draws(niter, time_dimension);
    Matrix second_harmonic_state_draws(niter, time_dimension);
    Vector sigma_trig_draws(niter);
    Vector sigma_obs_draws(niter);
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      const Matrix &state(model.state());
      first_harmonic_state_draws.row(i) = state.row(0);
      second_harmonic_state_draws.row(i) = state.row(2);
      sigma_trig_draws[i] = trig_state->error_distribution()->sigma();
      sigma_obs_draws[i] = model.observation_model()->sigma();
    }
    auto status = CheckMcmcMatrix(first_harmonic_state_draws,
                                  first_harmonic);
    EXPECT_TRUE(status.ok) << status;

    auto second_status = CheckMcmcMatrix(second_harmonic_state_draws,
                                         second_harmonic);
    EXPECT_TRUE(second_status.ok) << second_status;
  }

  
}  // namespace

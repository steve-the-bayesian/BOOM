#include "gtest/gtest.h"

#include "test_utils/test_utils.hpp"

#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"

#include "Models/StateSpace/MultivariateStateSpaceModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/MultivariateStateSpaceModelSampler.hpp"
#include "distributions.hpp"
#include "LinAlg/Array.hpp"

namespace {

  using namespace BOOM;
  using std::endl;

  class MultivariateStateSpaceModelTest : public ::testing::Test {
   protected:
    MultivariateStateSpaceModelTest() {
      GlobalRng::rng.seed(8675309);
    }

    // Generate fake parameters, and simulate the state and observed data.
    // Args:
    //   time_dimension:  The number of time points to simulate.
    //   ydim:  The dimension of the response to simulate.
    //   nfactors:  The number of factors in the state.
    void McmcSetup(int time_dimension, int ydim, int nfactors) {
      if (nfactors >= ydim) {
        report_error("The number of factors should be less than ydim.");
      }
      Vector state = rnorm_vector(nfactors, 0, 1);
      state_.resize(nfactors, time_dimension);
      observed_data_.resize(time_dimension, ydim);
      observation_coefficients_.resize(ydim, nfactors);
      observation_coefficients_.randomize();
      for (int i = 0; i < observation_coefficients_.nrow(); ++i) {
        for (int j = 0; j < std::min<int>(i, observation_coefficients_.ncol()); ++j) {
          observation_coefficients_(i, j) = 0.0;
        }
        if (i < observation_coefficients_.ncol()) {
          observation_coefficients_(i, i) = 1.0;
        }
      }
      
      for (int i = 0; i < time_dimension; ++i) {
        state += rnorm_vector(nfactors, 0, 1);
        state_.col(i) = state;
        observed_data_.row(i) = observation_coefficients_ * state
            + rnorm_vector(ydim, 0, 1);
      }
    }

    Matrix observed_data_;
    Matrix state_;
    Matrix observation_coefficients_;
  };

  //===========================================================================
  TEST_F(MultivariateStateSpaceModelTest, EmptyTest) {}

  //===========================================================================
  TEST_F(MultivariateStateSpaceModelTest, ConstructorTest) {
    MultivariateStateSpaceModel model(3);
  }

  //===========================================================================
  TEST_F(MultivariateStateSpaceModelTest, BaseClassTest) {
    MultivariateStateSpaceModel model(3);

    IndependentMvnModel *obs = model.observation_model();
    EXPECT_TRUE(obs != nullptr);
    EXPECT_EQ(obs->dim(), 3);
    EXPECT_EQ(0, model.number_of_state_models());
    EXPECT_EQ(0, model.state_dimension());
    EXPECT_EQ(0, model.time_dimension());
  }

  //===========================================================================
  TEST_F(MultivariateStateSpaceModelTest, McmcTest) {
    int time_dimension = 100;
    int ydim = 6;
    int nfactors = 2;
    int niter = 200;
    McmcSetup(time_dimension, ydim, nfactors);

    NEW(MultivariateStateSpaceModel, model)(ydim);
    for (int i = 0; i < observed_data_.nrow(); ++i) {
      NEW(PartiallyObservedVectorData, data_point)(observed_data_.row(i));
      model->add_data(data_point);
    }
    
    NEW(SharedLocalLevelStateModel, state_model)(nfactors, ydim, model.get());
    // Initial state mean and variance.
    Vector initial_state_mean(2, 0.0);
    SpdMatrix initial_state_variance(2, 1.0);
    state_model->set_initial_state_mean(initial_state_mean);
    state_model->set_initial_state_variance(initial_state_variance);

    // Prior distribution and posterior sampler.
    std::vector<Ptr<GammaModelBase>> innovation_precision_priors;
    for (int i = 0; i < nfactors; ++i) {
      innovation_precision_priors.push_back(new GammaModel(1, 1));
    }
    Matrix observation_coefficient_prior_mean(nfactors, ydim, 0.0);
    NEW(SharedLocalLevelPosteriorSampler, state_sampler)(
        state_model.get(),
        innovation_precision_priors,
        observation_coefficient_prior_mean,
        1.0);
    state_model->set_method(state_sampler);
    // Done configuring, so add the state model.
    model->add_state(state_model);

    // Check that the model matrices are as expected.
    int time_index = 2;
    Matrix transition = model->state_transition_matrix(time_index)->dense();
    EXPECT_TRUE(MatrixEquals(transition, SpdMatrix(2, 1.0)));

    state_model->coefficient_model()->set_Beta(observation_coefficients_.transpose());
    Matrix observation_coefficients = model->observation_coefficients(
        time_index, Selector(ydim, true))->dense();
    EXPECT_EQ(observation_coefficients.nrow(), observation_coefficients_.nrow());
    EXPECT_EQ(observation_coefficients.ncol(), observation_coefficients_.ncol());
    EXPECT_TRUE(MatrixEquals(observation_coefficients_, observation_coefficients))
        << endl << "correct observation matrix: " << endl
        << observation_coefficients_ << endl
        << "what the model has: " << endl
        << observation_coefficients;
    
    // Need a prior for the observation model.
    std::vector<Ptr<GammaModelBase>> observation_model_priors;
    for (int i = 0; i < ydim; ++i) {
      observation_model_priors.push_back(new GammaModel(1, 1));
    }
    NEW(IndependentMvnVarSampler, observation_model_sampler)(
        model->observation_model(),
        observation_model_priors);
    model->observation_model()->set_method(observation_model_sampler);

    // Set the global sampler for the model.
    NEW(MultivariateStateSpaceModelSampler, sampler)(model.get());
    model->set_method(sampler);
    
    Array state_draws({niter, model->state_dimension(), model->time_dimension()});
    Array observation_coefficient_draws({niter, ydim, model->state_dimension()});

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
    }
    
  }

  
}  // namespace

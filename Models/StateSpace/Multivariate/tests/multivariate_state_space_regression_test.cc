#include "gtest/gtest.h"

#include "test_utils/test_utils.hpp"

#include "Models/StateSpace/Multivariate/tests/mv_framework.hpp"

#include "cpputil/math_utils.hpp"

#include "Models/ChisqModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/PosteriorSamplers/IndependentMvnVarSampler.hpp"
#include "Models/PosteriorSamplers/ZeroMeanGaussianConjSampler.hpp"

#include "Models/Glm/MvnGivenX.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionSemiconjugateSampler.hpp"
#include "Models/Glm/PosteriorSamplers/IndependentRegressionModelsPosteriorSampler.hpp"

#include "Models/StateSpace/Multivariate/MultivariateStateSpaceRegressionModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/MvStateSpaceRegressionPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "distributions.hpp"
#include "LinAlg/Array.hpp"

namespace BoomStateSpaceTesting {

  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultivariateStateSpaceRegressionModelTest : public ::testing::Test {
   protected:
    MultivariateStateSpaceRegressionModelTest()
    {
      GlobalRng::rng.seed(8675310);
      GlobalRng::rng.seed(12345);
      GlobalRng::rng.seed(42);
    }
  };

  //===========================================================================
  std::vector<Vector> gather_state_specific_final_state(
      const MultivariateStateSpaceRegressionModel &model) {
    std::vector<Vector> ans(model.nseries());
    int last_time = model.time_dimension() - 1;
    if (last_time > 0) {
      for (int i = 0; i < ans.size(); ++i) {
        if (model.series_specific_model(i)->state_dimension() > 0) {
          ans[i] = model.series_specific_model(i)->state().col(last_time);
        }
      }
    }
    return ans;
  }

  //===========================================================================
  // Repeat each row of 'mat' for 'times' times, then move to the next row.
  Matrix repeat_rows(const Matrix &mat, int times) {
    Matrix ans(mat.nrow() * times, mat.ncol());
    int index = 0;
    for (int i = 0; i < mat.nrow(); ++i) {
      for (int j = 0; j < times; ++j) {
        ans.row(index++) = mat.row(i);
      }
    }
    return ans;
  }

  //===========================================================================
  TEST_F(MultivariateStateSpaceRegressionModelTest, EmptyTest) {}

  //===========================================================================
  TEST_F(MultivariateStateSpaceRegressionModelTest, ConstructorTest) {
    MultivariateStateSpaceRegressionModel model(3, 4);
  }

  //===========================================================================
  TEST_F(MultivariateStateSpaceRegressionModelTest, DataTest) {
    MultivariateTimeSeriesRegressionData data_point(3.2, Vector{1, 2, 3}, 0, 4);
    EXPECT_DOUBLE_EQ(3.2, data_point.y());
    EXPECT_TRUE(VectorEquals(Vector{1, 2, 3}, data_point.x()));
    EXPECT_EQ(0, data_point.series());
    EXPECT_EQ(4, data_point.timestamp());
  }

  //===========================================================================
  // Check that data objects contained in the model are the right size.
  TEST_F(MultivariateStateSpaceRegressionModelTest, ModelTest) {
    int ydim = 4;
    int xdim = 3;

    MultivariateStateSpaceRegressionModel model(xdim, ydim);
    EXPECT_EQ(0, model.state_dimension());
    EXPECT_EQ(0, model.number_of_state_models());
    EXPECT_EQ(nullptr, model.state_model(0));
    EXPECT_EQ(nullptr, model.state_model(-1));
    EXPECT_EQ(nullptr, model.state_model(2));
    EXPECT_EQ(0, model.time_dimension());

    EXPECT_EQ(ydim, model.nseries());
    EXPECT_EQ(xdim, model.xdim());

    std::vector<Ptr<MultivariateTimeSeriesRegressionData>> data;
    Matrix response_data(ydim, 12);
    for (int time = 0; time < 12; ++time) {
      for (int series = 0; series < ydim; ++series){
        NEW(MultivariateTimeSeriesRegressionData, data_point)(
            rnorm(0, 1), rnorm_vector(xdim, 0, 1), series, time);
        data.push_back(data_point);
        model.add_data(data_point);
        response_data(series, time) = data_point->y();
      }
    }
    EXPECT_EQ(12, model.time_dimension());
    for (int time = 0; time < 12; ++time) {
      for (int series = 0; series < ydim; ++series) {
        EXPECT_TRUE(model.is_observed(series, time));
        EXPECT_DOUBLE_EQ(response_data(series, time),
                         model.observed_data(series, time));
      }
    }

    int time = 27;
    int series = 2;
    NEW(MultivariateTimeSeriesRegressionData, data_point)(
        3.8, rnorm_vector(xdim, 0, 1), series, time);
    model.add_data(data_point);
    // The time dimension is one more than the largest time stamp on a data
    // point.  I.e. if times 0, 1, 2, 4 are observed then there are 5 time
    // points (with timestamp 3 being missing).
    EXPECT_EQ(model.time_dimension(), time + 1);
  }

  //===========================================================================
  // Check that adjusted_observation produces the expected result.  The
  // adjusted_observation should be the original observation minus the
  // regression effect.
  TEST_F(MultivariateStateSpaceRegressionModelTest, AdjustedObservationTest) {
    int xdim = 3;
    int nseries = 6;
    int nfactors = 1;
    int sample_size = 20;
    int test_size = 2;
    double residual_sd = .1;

    McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
                          test_size, residual_sd);

    for (int i = 0; i < nseries; ++i) {
      sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
      sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
    }
    //    sim.model->isolate_shared_state();

    sim.model->isolate_shared_state();
    Vector adjusted = sim.model->adjusted_observation(3);
    Vector regression_effect = sim.regression_coefficients * sim.predictors.row(3);
    Vector observed = sim.response.row(3);

    EXPECT_TRUE(VectorEquals(observed, adjusted + regression_effect))
        << "observed data at time 3: \n" << observed
        << "\nadjusted_observation: \n" << adjusted
        << "\nregression_effect:  \n" << regression_effect
        << "\ndifference: \n"
        << observed - adjusted - regression_effect;
  }

  //===========================================================================
  void set_observation_coefficients(
      const Matrix &coefficients,
      ConditionallyIndependentSharedLocalLevelStateModel &model) {
    for (int i = 0; i < coefficients.nrow(); ++i) {
      model.raw_observation_coefficients(i)->set_Beta(coefficients.row(i));
    }
  }

  //===========================================================================
  // Check if the state can be recovered if there is only one factor and (b) the
  // model parameters are known.  This is a precursor to MCMC success.
  // TEST_F(MultivariateStateSpaceRegressionModelTest, DrawStateTest) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 200;
  //   int test_size = 20;
  //   double residual_sd = .1;
  //   McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                         test_size, residual_sd);

  //   sim.model->observation_model()->clear_methods();

  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
  //   }

  //   sim.model->state_model(0)->clear_methods();
  //   set_observation_coefficients(sim.observation_coefficients, *sim.state_model);

  //   EXPECT_EQ(1, sim.model->state_dimension());

  //   int niter = 100;
  //   int burn = 10;
  //   Matrix state_draws(niter, sample_size);
  //   for (int i = 0; i < burn; ++i) {
  //     sim.model->sample_posterior();
  //     EXPECT_EQ(sim.model->shared_state().nrow(), 1);
  //     EXPECT_EQ(sim.model->shared_state().ncol(), sample_size);
  //   }
  //   for (int i = 0; i < niter; ++i) {
  //     sim.model->sample_posterior();
  //     state_draws.row(i) = sim.model->shared_state().row(0);
  //   }
  //   ConstVectorView true_state(sim.state.row(0), 0, sample_size);
  //   auto status = CheckMcmcMatrix(state_draws, true_state);
  //   EXPECT_TRUE(status.ok) << status;

  //   EXPECT_EQ("", CheckStochasticProcess(state_draws, true_state));

  //   std::ofstream state_draws_fixed_params("state_draws_fixed_params.out");
  //   state_draws_fixed_params << true_state << "\n"
  //                            << state_draws;

  // }

  //===========================================================================
  // Check draws of the state parameters given observation model parameters,
  // with the state fixed at true values.
  TEST_F(MultivariateStateSpaceRegressionModelTest, CheckStateParameterDraws) {
    int xdim = 3;
    int nseries = 2;
    int nfactors = 1;
    int sample_size = 200;
    int test_size = 20;
    double residual_sd = .1;

    McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
                          test_size, residual_sd);
    int niter = 100;
    int burn = 10;
    sim.model->observation_model()->clear_methods();
    for (int i = 0; i < nseries; ++i) {
      sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
      sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
    }

    sim.model->permanently_set_state(SubMatrix(
        sim.state, 0, nfactors - 1, 0, sample_size - 1).to_matrix());

    Matrix observation_coefficient_draws(niter, nseries);

    Matrix state_draws(niter, sample_size);
    for (int i = 0; i < burn; ++i) {
      sim.model->sample_posterior();
    }
    for (int i = 0; i < niter; ++i) {
      sim.model->sample_posterior();
      Vector stacked_beta(nseries);
      for (int j = 0; j < nseries; ++j) {
        stacked_beta[j] =
            sim.state_model->raw_observation_coefficients(j)->Beta()[0];
      }
      observation_coefficient_draws.row(i) = stacked_beta;
    }

    auto status = CheckMcmcMatrix(observation_coefficient_draws,
                                  sim.observation_coefficients.col(0));
    EXPECT_TRUE(status.ok) << status;

    std::ofstream obs_coef_full_conditional("obs_coef_full_conditional.out");
    obs_coef_full_conditional
        << sim.observation_coefficients.col(0) << "\n"
        << observation_coefficient_draws;
  }

  //===========================================================================
  // Check draws of the regression model parameters given the state model
  // parameters, with the state fixed at true values.
  TEST_F(MultivariateStateSpaceRegressionModelTest, CheckRegressionParameterDraws) {
    int xdim = 3;
    int nseries = 2;
    int nfactors = 1;
    int sample_size = 500;
    int test_size = 2;
    double residual_sd = .1;

    McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
                          test_size, residual_sd);
    int niter = 200;
    int burn = 10;
    sim.state_model->clear_methods();
    set_observation_coefficients(sim.observation_coefficients,
                                 *sim.state_model);

    sim.model->permanently_set_state(SubMatrix(
        sim.state, 0, nfactors - 1, 0, sample_size - 1).to_matrix());

    std::vector<Matrix> regression_coefficient_draws;
    for (int series = 0; series < nseries; ++series) {
      Matrix draws(niter, xdim);
      regression_coefficient_draws.push_back(draws);
    }
    Matrix residual_sd_draws(niter, nseries);

    Matrix state_draws(niter, sample_size);
    for (int i = 0; i < burn; ++i) {
      sim.model->sample_posterior();
    }
    for (int i = 0; i < niter; ++i) {
      sim.model->sample_posterior();
      for (int series = 0; series < nseries; ++series) {
        regression_coefficient_draws[series].row(i) =
            sim.model->observation_model()->model(series)->Beta();
        residual_sd_draws(i, series) =
            sim.model->observation_model()->model(series)->sigma();
      }
    }

    auto status = CheckMcmcMatrix(residual_sd_draws, Vector(nseries, residual_sd), 0.99);
    EXPECT_TRUE(status.ok) << "Residual SD did not cover true values " << status;
    std::ofstream rsd_draws("residual_sd_full_conditional.out");
    rsd_draws << Vector(nseries, residual_sd) << "\n" << residual_sd_draws;

    for (int series = 0; series < nseries; ++series) {
      status = CheckMcmcMatrix(regression_coefficient_draws[series],
                               sim.regression_coefficients.row(series));
      std::ostringstream fname;
      fname << "reg_coef_full_conditional_" << series << ".out";
      std::ofstream reg_coef_full_conditional(fname.str());
      reg_coef_full_conditional
          << sim.regression_coefficients.row(series) << "\n"
          << regression_coefficient_draws[series];

      EXPECT_TRUE(status.ok) << "Coefficients did not cover for series "
                             << series << ".  " << status
                             << "\n"
                             << "Check in " << fname.str() << "\n\n";
    }
  }

  TEST_F(MultivariateStateSpaceRegressionModelTest, TestFilter) {
    int xdim = 3;
    int nseries = 10;
    int nfactors = 1;
    int sample_size = 80;
    int test_size = 20;
    double residual_sd = .1;

    McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
                          test_size, residual_sd);

    sim.model->observation_model()->clear_methods();
    for (int i = 0; i < nseries; ++i) {
      sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
      sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
    }

    set_observation_coefficients(sim.observation_coefficients,
                                 *sim.state_model);
    sim.state_model->clear_methods();

    // Get off the initial conditions.  This is probably unnecessary.
    for (int i = 0; i < 3; ++i) {
      sim.model->sample_posterior();
    }

    // Run the sparse filter and the dense filter.
    sim.model->get_filter().update();
    BoomStateSpaceTesting::MockKalmanFilter dense_filter(sim.model.get());
    dense_filter.set_initial_state_mean(sim.state_model->initial_state_mean());
    dense_filter.set_initial_state_variance(sim.state_model->initial_state_variance());
    dense_filter.filter();

    // Check that a few relevant quantities are equal.
    int index = 3;
    EXPECT_TRUE(VectorEquals(sim.model->get_filter()[index].state_mean(),
                             dense_filter[index].state_mean()))
        << "State means were not equal for node " << index << ": \n"
        << "   sparse: \n"
        << sim.model->get_filter()[index].state_mean() << "\n"
        << "   dense:\n"
        << dense_filter[index].state_mean();

    index = sample_size - 1;
    EXPECT_TRUE(VectorEquals(sim.model->get_filter()[index].state_mean(),
                             dense_filter[index].state_mean()))
        << "State means were not equal for node " << index << ": \n"
        << "   sparse: \n"
        << sim.model->get_filter()[index].state_mean() << "\n"
        << "   dense:\n"
        << dense_filter[index].state_mean();


    EXPECT_TRUE(VectorEquals(sim.model->get_filter()[3].state_mean(),
                             dense_filter[3].state_mean()))
        << "State means were not equal for node 3: \n"
        << "   sparse: \n"
        << sim.model->get_filter()[3].state_mean() << "\n"
        << "   dense:\n"
        << dense_filter[3].state_mean();

    EXPECT_TRUE(MatrixEquals(sim.model->get_filter()[5].state_variance(),
                             dense_filter[5].state_variance()));

    Selector fully_observed(nseries, true);
    Ptr<SparseKalmanMatrix> forecast_precision =
        sim.model->get_filter()[5].sparse_forecast_precision();
    EXPECT_TRUE(MatrixEquals(
        sim.model->get_filter()[5].sparse_kalman_gain(
            fully_observed, forecast_precision)->dense(),
        dense_filter[5].kalman_gain(fully_observed)));

    forecast_precision = sim.model->get_filter()[
        sample_size - 1].sparse_forecast_precision();
    EXPECT_TRUE(MatrixEquals(
        sim.model->get_filter()[sample_size - 1].sparse_kalman_gain(
            fully_observed, forecast_precision)->dense(),
        dense_filter[sample_size - 1].kalman_gain(fully_observed)));

    EXPECT_TRUE(MatrixEquals(
        sim.model->get_filter()[5].sparse_forecast_precision()->dense(),
        dense_filter[5].forecast_precision()));

    EXPECT_TRUE(MatrixEquals(
        sim.model->get_filter()[sample_size - 1].sparse_forecast_precision()->dense(),
        dense_filter[sample_size - 1].forecast_precision()));

    //-------------------------------------------------------------------------
    // The tests above pass.  There is no meaningful numerical error in the
    // forward version of the filter.
    // -------------------------------------------------------------------------

    // Run the filter backwards and check that the smooths are the same.
    sim.model->get_filter().fast_disturbance_smooth();
    dense_filter.fast_disturbance_smooth();

    index = sample_size - 1;
    EXPECT_TRUE(VectorEquals(
        sim.model->get_filter()[index].scaled_state_error(),
        dense_filter[index].scaled_state_error()))
        << "Values of 'r' disagree in position " << index << ": \n"
        << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
        << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();

    index = sample_size - 2;
    EXPECT_TRUE(VectorEquals(
        sim.model->get_filter()[index].scaled_state_error(),
        dense_filter[index].scaled_state_error()))
        << "Values of 'r' disagree in position " << index << ": \n"
        << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
        << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();

    index = sample_size - 10;
    EXPECT_TRUE(VectorEquals(
        sim.model->get_filter()[index].scaled_state_error(),
        dense_filter[index].scaled_state_error()))
        << "Values of 'r' disagree in position " << index << ": \n"
        << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
        << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();

    index = 5;
    EXPECT_TRUE(VectorEquals(
        sim.model->get_filter()[index].scaled_state_error(),
        dense_filter[index].scaled_state_error()))
        << "Values of 'r' disagree in position " << index << ": \n"
        << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
        << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();
  }

  //===========================================================================
  // See how the multivariate results stack up vs an equivalent scalar model.
  TEST_F(MultivariateStateSpaceRegressionModelTest, ScalarComparisonTest) {
    int xdim = 3;
    int nseries = 1;
    int nfactors = 1;
    int sample_size = 100;
    int test_size = 2;
    double residual_sd = .1;

    McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
                          test_size, residual_sd);
    sim.regression_coefficients(0, 1) = 100.0;
    sim.build(residual_sd);

    // ---------------------------------------------------------------------------
    // Build the scalar model
    // ---------------------------------------------------------------------------
    NEW(StateSpaceRegressionModel, scalar_model)(xdim);
    for (int i = 0; i < sim.response.nrow(); ++i) {
      NEW(RegressionData, scalar_dp)(sim.response(i, 0), sim.predictors.row(i));
      scalar_model->add_regression_data(scalar_dp);
    }

    scalar_model->observation_model()->set_Beta(sim.regression_coefficients.row(0));
    scalar_model->observation_model()->set_sigsq(square(residual_sd));

    // TODO: multiply factor_sd by the first observation coefficient.
    NEW(LocalLevelStateModel, scalar_trend)(1.0);
    // set initial state distribution.
    scalar_model->add_state(scalar_trend);

    // ---------------------------------------------------------------------------
    // Build the multivariate model
    // ---------------------------------------------------------------------------
    sim.model->observation_model()->clear_methods();
    for (int i = 0; i < nseries; ++i) {
      sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
      sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
    }

    set_observation_coefficients(sim.observation_coefficients,
                                 *sim.state_model);
    sim.state_model->clear_methods();

    int niter = 100;

    Matrix scalar_state_draws(niter, sample_size);
    Matrix mv_state_draws(niter, sample_size);
    for (int i = 0; i < niter; ++i) {
      scalar_model->impute_state(GlobalRng::rng);
      // sim.model->impute_state(GlobalRng::rng);
      sim.model->sample_posterior();
      scalar_state_draws.row(i) = ConstVectorView(
          scalar_model->state_contribution(0), 0, sample_size);
      mv_state_draws.row(i) = ConstVectorView(
          sim.model->state_contributions(0).row(0),
          0, sample_size);
    }

    std::ofstream scalar_draws_file("scalar_state_draws.out");
    scalar_draws_file << ConstVectorView(sim.state.row(0), 0, sample_size) << "\n"
                      << scalar_state_draws;

    std::ofstream mv_state_draws_file("mv_state_draws.out");
    mv_state_draws_file << ConstVectorView(sim.state.row(0), 0, sample_size)
                        << "\n" << mv_state_draws;
  }

  //===========================================================================
  // Test the full MCMC experience.
  TEST_F(MultivariateStateSpaceRegressionModelTest, McmcTest) {
    // Simulate fake data from the model: shared local level and a regression
    // effect.
    int xdim = 3;
    int nseries = 10;
    int nfactors = 1;
    int sample_size = 100;
    int test_size = 20;
    double residual_sd = .1;

    McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
                          test_size, residual_sd);
    sim.regression_coefficients(0, 1) = 100.0;
    sim.build(residual_sd);

    EXPECT_EQ(sim.xdim(), xdim);
    EXPECT_EQ(sim.nseries(), nseries);
    EXPECT_EQ(sim.regression_coefficients.nrow(), nseries);
    EXPECT_EQ(sim.regression_coefficients.ncol(), xdim);

    ofstream("mcmc_raw_data.out") << sim.response;
    ofstream("mcmc_predictors.out") << sim.predictors;
    ofstream("true_state_contributions.out") << sim.observation_coefficients * sim.state;
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Set parameters to their true values so we can isolate the MCMC to find
    // the bug.

    //    sim.model->observation_model()->clear_methods();
    for (int i = 0; i < nseries; ++i) {
      sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
      sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
    }

    set_observation_coefficients(sim.observation_coefficients,
                                 *sim.state_model);
    //    sim.state_model->clear_methods();

    // sim.model->permanently_set_state(SubMatrix(
    //     sim.state, 0, nfactors - 1, 0, sample_size - 1).to_matrix());

    // sim.state_model->clear_methods();
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    int niter = 1000;
    int burn = 300;

    //---------------------------------------------------------------------------
    // Create space to store various different kinds of MCMC draws.
    //---------------------------------------------------------------------------

    // Draws of the raw state variables.  These might not be identified.
    std::vector<Matrix> factor_draws;
    for (int i = 0; i < nfactors; ++i) {
      Matrix draws(niter, sample_size);
      factor_draws.push_back(draws);
    }

    // The state contribution is the observation coefficients times the factor
    // values.

    // stored on a series by series basis.
    std::vector<Matrix> state_contribution_draws(
        nseries, Matrix(niter, sample_size));

    // Store the output of calls to predict().
    Array prediction_draws(std::vector<int>{
        niter, sim.model->nseries(), test_size});

    // Store the observation coefficients.
    Array observation_coefficient_draws(
        std::vector<int>{niter, nseries, nfactors});

    Array regression_coefficient_draws(
        std::vector<int>{niter, nseries, xdim});

    // Draws of the residual standard deviation parameters.
    Matrix residual_sd_draws(niter, nseries);

    int state_error_dimension = nfactors;
    Matrix innovation_sd_draws(niter, state_error_dimension);

    // ---------------------------------------------------------------------------
    // Workspace for dealing with predictions.
    // ---------------------------------------------------------------------------

    // Predictor variables for the test set.
    Matrix test_predictors;
    if (test_size > 0) {
      test_predictors = ConstSubMatrix(
          sim.predictors,
          sample_size, sample_size + test_size - 1,
          0, ncol(sim.predictors) - 1).to_matrix();
      test_predictors = repeat_rows(test_predictors, nseries);
    }

    ofstream prediction_out("prediction.draws");
    prediction_out << ConstSubMatrix(sim.response, sample_size, sample_size + test_size - 1,
                                     0, nseries - 1).transpose();

    //---------------------------------------------------------------------------
    // Run the MCMC -- burn-in
    //---------------------------------------------------------------------------
    Selector fully_observed(test_size, true);
    sim.model->observe_time_dimension(sample_size + test_size);
    EXPECT_EQ(sim.model->time_dimension(), sample_size);
    for (int i = 0; i < burn; ++i) {
      sim.model->sample_posterior();
    }

    //---------------------------------------------------------------------------
    // Run the MCMC -- main algorithm
    //---------------------------------------------------------------------------
    for (int i = 0; i < niter; ++i) {
      if (i % (100) == 0) {
        cout << "------ draw " << i << " of " << niter << " ---------\n";
      }
      sim.model->sample_posterior();

      Matrix local_state_contribution_draw = sim.model->state_contributions(0);
      for (int series = 0; series < nseries; ++series) {
        state_contribution_draws[series].row(i) =
            local_state_contribution_draw.row(series);
      }

      for (int factor = 0; factor < nfactors; ++factor) {
        factor_draws[factor].row(i) = sim.model->shared_state().row(factor);
      }
      Matrix Z = sim.model->observation_coefficients(0, fully_observed)->dense();
      observation_coefficient_draws.slice(i, -1, -1) = Z;
      for (int series = 0; series < nseries; ++series) {
        residual_sd_draws(i, series) =
            sim.model->observation_model()->model(series)->sigma();
        regression_coefficient_draws.slice(i, series, -1) =
            sim.model->observation_model()->model(series)->Beta();
      }

      innovation_sd_draws.row(i) =
          sqrt(sim.model->state_error_variance(2)->dense().diag());

      if (test_size > 0) {
      prediction_draws.slice(i, -1, -1) = sim.model->simulate_forecast(
          GlobalRng::rng,
          test_predictors,
          sim.model->shared_state().last_col(),
          gather_state_specific_final_state(*sim.model));
      prediction_out << prediction_draws.slice(i, -1, -1).to_matrix();
      }
    }

    Vector residual_sd_vector(nseries, residual_sd);
    ofstream("residual_sd.draws") << residual_sd_vector << "\n"
                                  << residual_sd_draws;
    auto status = CheckMcmcMatrix(residual_sd_draws, residual_sd_vector);
    EXPECT_TRUE(status.ok) << "Problem with residual sd draws." << status;

    // Factor draws are not identified.  Factors * observation coefficients is
    // identified.
    //
    // This section prints out the (maybe unidentified) factor and observation
    // coefficient levels.
    for (int factor = 0; factor < nfactors; ++factor) {
      ConstVectorView true_factor(sim.state.row(factor), 0, sample_size);
      status = CheckMcmcMatrix(factor_draws[factor], true_factor);
      std::ostringstream fname;
      fname << "factor_" << factor << "_draws.out";
      ofstream factor_draws_out(fname.str());
      factor_draws_out << true_factor << "\n" << factor_draws[factor];

      std::ostringstream observation_coefficient_fname;
      observation_coefficient_fname
          << "observation_coefficient_draws_factor_" << factor;
      std::ofstream obs_coef_out(observation_coefficient_fname.str());
      obs_coef_out << sim.observation_coefficients.col(0) << "\n"
                   << observation_coefficient_draws.slice(-1, -1, 0);

      //      EXPECT_TRUE(status.ok) << "Error in factor " << factor << ".  " << status;
    }

    // Print out the state contribution to each series.  These should be
    // identified.
    //
    // Also print the regression coefficients for each series.

    Matrix true_state_contributions = sim.observation_coefficients * sim.state;
    for (int series = 0; series < nseries; ++series) {
      std::ostringstream fname;
      fname << "state_contribution_series_" << series;
      std::ofstream state_contribution_out(fname.str());
      ConstVectorView truth(true_state_contributions.row(series),
                            0, sample_size);
      state_contribution_out << truth << "\n"
                             << state_contribution_draws[series];
      double confidence = .99;
      double sd_ratio_threshold = 0.1;
      double coverage_fraction = .25;
      std::string error_message = CheckStochasticProcess(
          state_contribution_draws[series], truth, confidence,
          sd_ratio_threshold, coverage_fraction);
      EXPECT_EQ(error_message, "") << "Error in state contribution draws for series "
                             << series << ".  " << "\n"
                             << "See the draws in file " << fname.str()
                             << "\n\n";

      std::ostringstream reg_fname;
      reg_fname << "regression_coefficient_mcmc_draws_series_" << series;
      std::ofstream reg_out(reg_fname.str());
      reg_out << sim.regression_coefficients.row(series) << "\n"
              << regression_coefficient_draws.slice(-1, series, -1);
    }

    std::ofstream("state_error_sd_mcmc_draws.out")
        << Vector(nfactors, 1.0) << "\n" << innovation_sd_draws;
    std::cerr << "Low confidence in tests.  Some commented out.\n";
  }

  /*
    library(Boom)

    PlotDrawsTs <- function(fname, ...) {
      x <- mscan(fname)
      truth <- x[1, ]
      draws <- x[-1, ]
      PlotManyTs(draws, truth=truth, ylim = range(draws, truth), ...)
      return(invisible(x))
    }

    PlotDrawsDist <- function(fname, relative = FALSE, ...) {
      x <- mscan(fname)
      truth <- x[1, ]
      draws <- x[-1, ]
      if (relative) {
        draws <- t(t(draws) - truth)
          truth <- rep(0, length(truth))
      }

      PlotDynamicDistribution(draws, ylim = range(draws, truth), ...)
      lines(truth, col="green", lwd=3)
      return(invisible(x))
    }

    // R code for viewing the results.
    plot.predictions <- function(fname, nseries, burn = 0) {
      library(Boom)
      draws.mat <- mscan(fname)
      truth <- draws.mat[1:nseries, ]
      draws.mat <- draws.mat[-(1:nseries), ]
      niter <- nrow(draws.mat) / nseries
      ntimes <- ncol(draws.mat)
      draws <- aperm(array(draws.mat, dim = c(nseries, niter, ntimes)), c(2, 1, 3))
      if (burn > 0) {
        draws <- draws[-(1:burn), , ]
      }

      nr <- max(1, floor(sqrt(nseries)))
      nc <- ceiling(nseries / nr)
      opar <- par(mfrow = c(nr, nc))
      for (i in 1:nseries) {
      PlotDynamicDistribution(draws[, i, ], ylim = range(draws, truth))
      lines(truth[i, ], lty = 2, col = "green")
      }
    }

    plot.observation.coefs <- function() {
      library(Boom)
      coefs <- mscan("observation_coefficient.draws")
      BoxplotTrue(coefs[-1, ], truth = coefs[1, ])
    }

    f1 <- mscan("factor1.draws")
    PlotDynamicDistribution(f1[-1, ])
    lines(f1[1, ], col = "green")
  */

  //=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

  // //===========================================================================
  // // A test case with both shared state and a single series that has series
  // // specific state (in this case a seasonal model).
  // TEST_F(MultivariateStateSpaceRegressionModelTest, SharedPlusIndividualTest) {
  //   int xdim = 3;
  //   int nseries = 8;
  //   int nfactors = 3;
  //   int sample_size = 250;

  //   int special_series = 4;
  //   int nseasons = 7;
  //   double seasonal_innovation_sd = .2;
  //   double residual_sd = .1;

  //   //----------------------------------------------------------------------
  //   // Simulate the shared state.
  //   //----------------------------------------------------------------------
  //   Matrix state(nfactors, sample_size, 0.0);
  //   for (int factor = 0; factor < nfactors; ++factor) {
  //     state(factor, 0) = rnorm();
  //     for (int time = 1; time < sample_size; ++time) {
  //       state(factor, time) = state(factor, time - 1) + rnorm(0, 1.0);
  //     }
  //   }

  //   Matrix observation_coefficients(nseries, nfactors);
  //   observation_coefficients.randomize();
  //   observation_coefficients.diag() = 1.0;
  //   for (int i = 0; i < std::min<int>(nrow(observation_coefficients),
  //                                     ncol(observation_coefficients)); ++i) {
  //     for (int j = i + 1; j < ncol(observation_coefficients); ++j) {
  //       observation_coefficients(i, j) = 0.0;
  //     }
  //   }

  //   // The columns of state_contribution are time points.  The rows are series.
  //   Matrix state_contribution = (observation_coefficients * state).transpose();

  //   //----------------------------------------------------------------------
  //   // Simulate the regression component.
  //   //----------------------------------------------------------------------
  //   Matrix predictors(sample_size * nseries, xdim);
  //   predictors.randomize();
  //   predictors.col(0) = 1.0;

  //   Matrix regression_coefficients(nseries, xdim);
  //   regression_coefficients.randomize();

  //   //----------------------------------------------------------------------
  //   // Simulate a seasonal pattern for one of the series.
  //   //----------------------------------------------------------------------
  //   Vector seasonal_pattern = rnorm_vector(7 - 1, 0, 20);
  //   SeasonalStateSpaceMatrix seasonal_matrix(nseasons);
  //   Vector seasonal(sample_size);
  //   for (int i = 0; i < sample_size; ++i) {
  //     seasonal[i] = seasonal_pattern[0];
  //     seasonal_pattern = seasonal_matrix * ConstVectorView(seasonal_pattern);
  //     seasonal_pattern[0] += rnorm(0, seasonal_innovation_sd);
  //   }

  //   //----------------------------------------------------------------------
  //   // Simulate errors, and add them to get responses.
  //   //----------------------------------------------------------------------
  //   Vector errors = rnorm_vector(nseries * sample_size, 0, residual_sd);

  //   //==========================================================================
  //   // Construct the model
  //   //==========================================================================

  //   NEW(MultivariateStateSpaceRegressionModel, model)(xdim, nseries);

  //   //----------------------------------------------------------------------
  //   // Add data
  //   //----------------------------------------------------------------------
  //   int index = -1;
  //   for (int time = 0; time < sample_size; ++time) {
  //     for (int series = 0; series < nseries; ++series) {
  //       ++index;
  //       double regression = predictors.row(index).dot(
  //           regression_coefficients.row(series));
  //       double y = regression + errors[index]
  //           + observation_coefficients.row(series).dot(state.col(time));
  //       NEW(MultivariateTimeSeriesRegressionData, data_point)(
  //           y, predictors.row(index), series, time);
  //       model->add_data(data_point);
  //     }
  //   }

  //   //----------------------------------------------------------------------
  //   // Add state models
  //   //----------------------------------------------------------------------
  //   NEW(SharedLocalLevelStateModel, state_model)(nfactors, model.get(), nseries);

  //   // Add posterior sampler for state model.
  //   std::vector<Ptr<VariableSelectionPrior>> spikes;
  //   std::vector<Ptr<MvnBase>> slabs;
  //   for (int i = 0; i < model->nseries(); ++i) {
  //     // Inclusion probabilities will get adjusted in the constructor for the
  //     // posterior sampler.
  //     Vector inc_probs(nfactors, 1.0);
  //     NEW(VariableSelectionPrior, spike)(inc_probs);
  //     spikes.push_back(spike);

  //     NEW(MvnGivenXMvRegSuf, slab)(
  //         new VectorParams(Vector(nfactors, 1.0)),
  //         new UnivParams(1.0),
  //         Vector(),
  //         .5,
  //         state_model->coefficient_model()->suf());
  //     slabs.push_back(slab);
  //   }
  //   NEW(SharedLocalLevelPosteriorSampler, state_model_sampler)(
  //       state_model.get(), slabs, spikes);
  //   state_model->set_observation_coefficients(observation_coefficients);
  //   state_model->set_method(state_model_sampler);

  //   // Add the initial distribution for the state model.
  //   state_model->set_initial_state_mean(Vector(nfactors, 0.0));
  //   state_model->set_initial_state_variance(SpdMatrix(nfactors, 100.0));

  //   model->add_state(state_model);

  //   //----------------------------------------------------------------------
  //   // Add a series specific state model for the special series.
  //   //----------------------------------------------------------------------
  //   Ptr<SeasonalStateModel> seasonal_model(new SeasonalStateModel(nseasons, 1));
  //   seasonal_model->set_sigsq(square(seasonal_innovation_sd));

  //   seasonal_model->set_initial_state_mean(
  //       Vector(seasonal_model->state_dimension(), 0.0));
  //   seasonal_model->set_initial_state_variance(100);

  //   NEW(ZeroMeanGaussianConjSampler, seasonal_model_sampler)(
  //       seasonal_model.get(), 1, seasonal_innovation_sd);
  //   seasonal_model->set_method(seasonal_model_sampler);

  //   model->add_series_specific_state(seasonal_model, special_series);

  //   //----------------------------------------------------------------------
  //   // Add Samplers for the observation model
  //   //----------------------------------------------------------------------
  //   for (int series = 0; series < model->nseries(); ++series) {
  //     Ptr<RegressionModel> series_reg =
  //         model->observation_model()->model(series);
  //     model->observation_model()->model(series)->set_Beta(
  //         regression_coefficients.row(series));
  //     model->observation_model()->model(series)->set_sigsq(square(residual_sd));

  //     NEW(MvnGivenScalarSigma, slab)(Vector(xdim, 0.0),
  //                                    SpdMatrix(xdim, 1.0 / 10000.0),
  //                                    series_reg->Sigsq_prm());
  //     Vector prior_inclusion_probabilities(xdim, 0.5);
  //     NEW(VariableSelectionPrior, spike)(prior_inclusion_probabilities);
  //     NEW(ChisqModel, residual_precision_prior)(10000000, residual_sd);
  //     NEW(BregVsSampler, reg_sampler)(
  //         series_reg.get(), slab, residual_precision_prior, spike);
  //     series_reg->set_method(reg_sampler);
  //   }
  //   NEW(IndependentRegressionModelsPosteriorSampler, observation_model_sampler)(
  //       model->observation_model());
  //   model->observation_model()->set_method(observation_model_sampler);

  //   //---------------------------------------------------------------------------
  //   // Check that the proxy model has the correct model matrices.
  //   //---------------------------------------------------------------------------
  //   EXPECT_DOUBLE_EQ(square(residual_sd),
  //                    model->series_specific_model(
  //                        special_series)->observation_variance(2));

  //   StateSpaceModel temp_model;
  //   temp_model.add_state(seasonal_model);

  //   EXPECT_TRUE(MatrixEquals(
  //       temp_model.state_transition_matrix(2)->dense(),
  //       model->series_specific_model(
  //           special_series)->state_transition_matrix(2)->dense()));

  //   EXPECT_TRUE(MatrixEquals(
  //       temp_model.state_variance_matrix(2)->dense(),
  //       model->series_specific_model(
  //           special_series)->state_variance_matrix(2)->dense()));

  //   EXPECT_TRUE(MatrixEquals(
  //       temp_model.state_error_expander(2)->dense(),
  //       model->series_specific_model(
  //           special_series)->state_error_expander(2)->dense()));

  //   EXPECT_TRUE(MatrixEquals(
  //       temp_model.state_error_variance(2)->dense(),
  //       model->series_specific_model(
  //           special_series)->state_error_variance(2)->dense()));

  //   EXPECT_TRUE(VectorEquals(
  //       temp_model.observation_matrix(2).dense(),
  //       model->series_specific_model(
  //           special_series)->observation_matrix(2).dense()));

  //   EXPECT_NEAR(square(residual_sd),
  //               model->series_specific_model(
  //                   special_series)->observation_variance(2),
  //               1e-8);

  //   EXPECT_TRUE(MatrixEquals(
  //       model->series_specific_model(special_series)->initial_state_variance(),
  //       SpdMatrix(6, 100.0)));

  //   //----------------------------------------------------------------------
  //   // Add Sampler for the model.
  //   //----------------------------------------------------------------------
  //   NEW(MultivariateStateSpaceRegressionPosteriorSampler, sampler)(model.get());
  //   model->set_method(sampler);

  //   //==========================================================================
  //   // Do the simulation
  //   //==========================================================================
  //   int burn = 20;
  //   for (int i = 0; i < burn; ++i) {
  //     model->sample_posterior();
  //   }
  //   int niter = 100;
  //   Matrix seasonal_effect_draws(niter, sample_size);
  //   Array coefficient_draws(std::vector<int>{niter, nseries, xdim});
  //   Matrix residual_sd_draws(niter, nseries);
  //   Matrix innovation_sd_draws(niter, nfactors);
  //   Vector seasonal_sd_draws(niter);

  //   for (int i = 0; i < niter; ++i) {
  //     //      model->sample_posterior();
  //     model->impute_state(GlobalRng::rng);
  //     seasonal_effect_draws.row(i) =
  //         model->series_specific_model(special_series)->state().row(0);
  //     seasonal_sd_draws[i] = seasonal_model->sigma();
  //     for (int j = 0; j < model->nseries(); ++j) {
  //       coefficient_draws.vector_slice(i, j, -1) =
  //           model->observation_model()->model(j)->Beta();
  //       residual_sd_draws(i, j) =
  //           model->observation_model()->model(j)->sigma();
  //     }
  //   }

  //   for (int i = 0; i < sample_size; ++i) {
  //     NEW(StateSpace::MultiplexedDoubleData, data_point)();
  //     data_point->add_data(new DoubleData(
  //         seasonal[i]
  //         + errors[i * nseries + special_series]));
  //     temp_model.add_data(data_point);
  //   }
  //   NEW(StateSpacePosteriorSampler, temp_model_sampler)(&temp_model);
  //   temp_model.observation_model()->set_sigsq(square(residual_sd));
  //   temp_model.set_method(temp_model_sampler);
  //   Matrix temp_model_state_draws(niter, sample_size);
  //   for (int i = 0; i < niter; ++i) {
  //     temp_model.sample_posterior();
  //     temp_model_state_draws.row(i) = temp_model.state_contribution(0);
  //   }
  // }

}  // namespace

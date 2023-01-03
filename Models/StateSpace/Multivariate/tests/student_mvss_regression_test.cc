#include "gtest/gtest.h"

#include "test_utils/test_utils.hpp"

#include "Models/StateSpace/Multivariate/tests/student_regression_framework.hpp"

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

#include "Models/StateSpace/Multivariate/StudentMvssRegressionModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/StudentMvssPosteriorSampler.hpp"
#include "Models/StateSpace/PosteriorSamplers/StateSpacePosteriorSampler.hpp"

#include "distributions.hpp"
#include "LinAlg/Array.hpp"

namespace BoomStateSpaceTesting {

  using namespace BOOM;
  using std::endl;
  using std::cout;

  class StudentMvssRegressionTest : public ::testing::Test {
   protected:
    StudentMvssRegressionTest()
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
  TEST_F(StudentMvssRegressionTest, EmptyTest) {}

  //===========================================================================
  TEST_F(StudentMvssRegressionTest, ConstructorTest) {
    StudentMvssRegressionModel model(3, 4);
  }

  //===========================================================================
  TEST_F(StudentMvssRegressionTest, DataTest) {
    StudentMultivariateTimeSeriesRegressionData data_point(3.2, Vector{1, 2, 3}, 0, 4);
    EXPECT_DOUBLE_EQ(3.2, data_point.y());
    EXPECT_TRUE(VectorEquals(Vector{1, 2, 3}, data_point.x()));
    EXPECT_EQ(0, data_point.series());
    EXPECT_EQ(4, data_point.timestamp());

    EXPECT_DOUBLE_EQ(1.0, data_point.weight());
    data_point.set_weight(1.2);
    EXPECT_DOUBLE_EQ(1.2, data_point.weight());
  }

  // //===========================================================================
  // // Check that adjusted_observation produces the expected result.  The
  // // adjusted_observation should be the original observation minus the
  // // regression effect.
  // TEST_F(MultivariateStateSpaceRegressionModelTest, AdjustedObservationTest) {
  //   int xdim = 3;
  //   int nseries = 6;
  //   int nfactors = 1;
  //   int sample_size = 20;
  //   int test_size = 2;
  //   double residual_sd = .1;

  //   McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                         test_size, residual_sd);

  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
  //   }
  //   //    sim.model->isolate_shared_state();

  //   sim.model->isolate_shared_state();
  //   Vector adjusted = sim.model->adjusted_observation(3);
  //   Vector regression_effect = sim.regression_coefficients * sim.predictors.row(3);
  //   Vector observed = sim.response.row(3);

  //   EXPECT_TRUE(VectorEquals(observed, adjusted + regression_effect))
  //       << "observed data at time 3: \n" << observed
  //       << "\nadjusted_observation: \n" << adjusted
  //       << "\nregression_effect:  \n" << regression_effect
  //       << "\ndifference: \n"
  //       << observed - adjusted - regression_effect;
  // }

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
  TEST_F(StudentMvssRegressionTest, DrawStateTest) {
    int xdim = 3;
    int nseries = 2;
    int nfactors = 1;
    int sample_size = 200;
    int test_size = 20;
    double residual_sd = .1;
    double tail_thickness = 3.0;
    StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
                             test_size, residual_sd, tail_thickness);

    sim.model->observation_model()->clear_methods();

    for (int i = 0; i < nseries; ++i) {
      sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
      sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
    }

    sim.model->state_model(0)->clear_methods();
    set_observation_coefficients(sim.observation_coefficients, *sim.state_model);

    EXPECT_EQ(1, sim.model->state_dimension());

    int niter = 100;
    int burn = 10;
    Matrix state_draws(niter, sample_size);
    for (int i = 0; i < burn; ++i) {
      sim.model->sample_posterior();
      EXPECT_EQ(sim.model->shared_state().nrow(), 1);
      EXPECT_EQ(sim.model->shared_state().ncol(), sample_size);
    }
    for (int i = 0; i < niter; ++i) {
      sim.model->sample_posterior();
      state_draws.row(i) = sim.model->shared_state().row(0);
    }
    ConstVectorView true_state(sim.state.row(0), 0, sample_size);
    auto status = CheckMcmcMatrix(state_draws, true_state);
    EXPECT_TRUE(status.ok) << status;

    EXPECT_EQ("", CheckStochasticProcess(state_draws, true_state));

    std::ofstream state_draws_fixed_params("state_draws_fixed_params.out");
    state_draws_fixed_params << true_state << "\n"
                             << state_draws;

  }

  // //===========================================================================
  // // Check draws of the state parameters given observation model parameters,
  // // with the state fixed at true values.
  // TEST_F(MultivariateStateSpaceRegressionModelTest, CheckStateParameterDraws) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 200;
  //   int test_size = 20;
  //   double residual_sd = .1;

  //   McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                         test_size, residual_sd);
  //   int niter = 100;
  //   int burn = 10;
  //   sim.model->observation_model()->clear_methods();
  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
  //   }

  //   sim.model->permanently_set_state(SubMatrix(
  //       sim.state, 0, nfactors - 1, 0, sample_size - 1).to_matrix());

  //   Matrix observation_coefficient_draws(niter, nseries);

  //   Matrix state_draws(niter, sample_size);
  //   for (int i = 0; i < burn; ++i) {
  //     sim.model->sample_posterior();
  //   }
  //   for (int i = 0; i < niter; ++i) {
  //     sim.model->sample_posterior();
  //     Vector stacked_beta(nseries);
  //     for (int j = 0; j < nseries; ++j) {
  //       stacked_beta[j] =
  //           sim.state_model->raw_observation_coefficients(j)->Beta()[0];
  //     }
  //     observation_coefficient_draws.row(i) = stacked_beta;
  //   }

  //   auto status = CheckMcmcMatrix(observation_coefficient_draws,
  //                                 sim.observation_coefficients.col(0));
  //   EXPECT_TRUE(status.ok) << status;

  //   std::ofstream obs_coef_full_conditional("obs_coef_full_conditional.out");
  //   obs_coef_full_conditional
  //       << sim.observation_coefficients.col(0) << "\n"
  //       << observation_coefficient_draws;
  // }

  // //===========================================================================
  // // Check draws of the regression model parameters given the state model
  // // parameters, with the state fixed at true values.
  // TEST_F(MultivariateStateSpaceRegressionModelTest, CheckRegressionParameterDraws) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 500;
  //   int test_size = 2;
  //   double residual_sd = .1;

  //   McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                         test_size, residual_sd);
  //   int niter = 200;
  //   int burn = 10;
  //   sim.state_model->clear_methods();
  //   set_observation_coefficients(sim.observation_coefficients,
  //                                *sim.state_model);

  //   sim.model->permanently_set_state(SubMatrix(
  //       sim.state, 0, nfactors - 1, 0, sample_size - 1).to_matrix());

  //   std::vector<Matrix> regression_coefficient_draws;
  //   for (int series = 0; series < nseries; ++series) {
  //     Matrix draws(niter, xdim);
  //     regression_coefficient_draws.push_back(draws);
  //   }
  //   Matrix residual_sd_draws(niter, nseries);

  //   Matrix state_draws(niter, sample_size);
  //   for (int i = 0; i < burn; ++i) {
  //     sim.model->sample_posterior();
  //   }
  //   for (int i = 0; i < niter; ++i) {
  //     sim.model->sample_posterior();
  //     for (int series = 0; series < nseries; ++series) {
  //       regression_coefficient_draws[series].row(i) =
  //           sim.model->observation_model()->model(series)->Beta();
  //       residual_sd_draws(i, series) =
  //           sim.model->observation_model()->model(series)->sigma();
  //     }
  //   }

  //   auto status = CheckMcmcMatrix(residual_sd_draws, Vector(nseries, residual_sd), 0.99);
  //   EXPECT_TRUE(status.ok) << "Residual SD did not cover true values " << status;
  //   std::ofstream rsd_draws("residual_sd_full_conditional.out");
  //   rsd_draws << Vector(nseries, residual_sd) << "\n" << residual_sd_draws;

  //   for (int series = 0; series < nseries; ++series) {
  //     status = CheckMcmcMatrix(regression_coefficient_draws[series],
  //                              sim.regression_coefficients.row(series));
  //     std::ostringstream fname;
  //     fname << "reg_coef_full_conditional_" << series << ".out";
  //     std::ofstream reg_coef_full_conditional(fname.str());
  //     reg_coef_full_conditional
  //         << sim.regression_coefficients.row(series) << "\n"
  //         << regression_coefficient_draws[series];

  //     EXPECT_TRUE(status.ok) << "Coefficients did not cover for series "
  //                            << series << ".  " << status
  //                            << "\n"
  //                            << "Check in " << fname.str() << "\n\n";
  //   }
  // }

  // TEST_F(MultivariateStateSpaceRegressionModelTest, TestFilter) {
  //   int xdim = 3;
  //   int nseries = 10;
  //   int nfactors = 1;
  //   int sample_size = 80;
  //   int test_size = 20;
  //   double residual_sd = .1;

  //   McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                         test_size, residual_sd);

  //   sim.model->observation_model()->clear_methods();
  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
  //   }

  //   set_observation_coefficients(sim.observation_coefficients,
  //                                *sim.state_model);
  //   sim.state_model->clear_methods();

  //   // Get off the initial conditions.  This is probably unnecessary.
  //   for (int i = 0; i < 3; ++i) {
  //     sim.model->sample_posterior();
  //   }

  //   // Run the sparse filter and the dense filter.
  //   sim.model->get_filter().update();
  //   BoomStateSpaceTesting::MockKalmanFilter dense_filter(sim.model.get());
  //   dense_filter.set_initial_state_mean(sim.state_model->initial_state_mean());
  //   dense_filter.set_initial_state_variance(sim.state_model->initial_state_variance());
  //   dense_filter.filter();

  //   // Check that a few relevant quantities are equal.
  //   int index = 3;
  //   EXPECT_TRUE(VectorEquals(sim.model->get_filter()[index].state_mean(),
  //                            dense_filter[index].state_mean()))
  //       << "State means were not equal for node " << index << ": \n"
  //       << "   sparse: \n"
  //       << sim.model->get_filter()[index].state_mean() << "\n"
  //       << "   dense:\n"
  //       << dense_filter[index].state_mean();

  //   index = sample_size - 1;
  //   EXPECT_TRUE(VectorEquals(sim.model->get_filter()[index].state_mean(),
  //                            dense_filter[index].state_mean()))
  //       << "State means were not equal for node " << index << ": \n"
  //       << "   sparse: \n"
  //       << sim.model->get_filter()[index].state_mean() << "\n"
  //       << "   dense:\n"
  //       << dense_filter[index].state_mean();


  //   EXPECT_TRUE(VectorEquals(sim.model->get_filter()[3].state_mean(),
  //                            dense_filter[3].state_mean()))
  //       << "State means were not equal for node 3: \n"
  //       << "   sparse: \n"
  //       << sim.model->get_filter()[3].state_mean() << "\n"
  //       << "   dense:\n"
  //       << dense_filter[3].state_mean();

  //   EXPECT_TRUE(MatrixEquals(sim.model->get_filter()[5].state_variance(),
  //                            dense_filter[5].state_variance()));

  //   Selector fully_observed(nseries, true);
  //   Ptr<SparseKalmanMatrix> forecast_precision =
  //       sim.model->get_filter()[5].sparse_forecast_precision();
  //   EXPECT_TRUE(MatrixEquals(
  //       sim.model->get_filter()[5].sparse_kalman_gain(
  //           fully_observed, forecast_precision)->dense(),
  //       dense_filter[5].kalman_gain(fully_observed)));

  //   forecast_precision = sim.model->get_filter()[
  //       sample_size - 1].sparse_forecast_precision();
  //   EXPECT_TRUE(MatrixEquals(
  //       sim.model->get_filter()[sample_size - 1].sparse_kalman_gain(
  //           fully_observed, forecast_precision)->dense(),
  //       dense_filter[sample_size - 1].kalman_gain(fully_observed)));

  //   EXPECT_TRUE(MatrixEquals(
  //       sim.model->get_filter()[5].sparse_forecast_precision()->dense(),
  //       dense_filter[5].forecast_precision()));

  //   EXPECT_TRUE(MatrixEquals(
  //       sim.model->get_filter()[sample_size - 1].sparse_forecast_precision()->dense(),
  //       dense_filter[sample_size - 1].forecast_precision()));

  //   //-------------------------------------------------------------------------
  //   // The tests above pass.  There is no meaningful numerical error in the
  //   // forward version of the filter.
  //   // -------------------------------------------------------------------------

  //   // Run the filter backwards and check that the smooths are the same.
  //   sim.model->get_filter().fast_disturbance_smooth();
  //   dense_filter.fast_disturbance_smooth();

  //   index = sample_size - 1;
  //   EXPECT_TRUE(VectorEquals(
  //       sim.model->get_filter()[index].scaled_state_error(),
  //       dense_filter[index].scaled_state_error()))
  //       << "Values of 'r' disagree in position " << index << ": \n"
  //       << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
  //       << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();

  //   index = sample_size - 2;
  //   EXPECT_TRUE(VectorEquals(
  //       sim.model->get_filter()[index].scaled_state_error(),
  //       dense_filter[index].scaled_state_error()))
  //       << "Values of 'r' disagree in position " << index << ": \n"
  //       << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
  //       << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();

  //   index = sample_size - 10;
  //   EXPECT_TRUE(VectorEquals(
  //       sim.model->get_filter()[index].scaled_state_error(),
  //       dense_filter[index].scaled_state_error()))
  //       << "Values of 'r' disagree in position " << index << ": \n"
  //       << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
  //       << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();

  //   index = 5;
  //   EXPECT_TRUE(VectorEquals(
  //       sim.model->get_filter()[index].scaled_state_error(),
  //       dense_filter[index].scaled_state_error()))
  //       << "Values of 'r' disagree in position " << index << ": \n"
  //       << "   sparse: " << sim.model->get_filter()[index].scaled_state_error()
  //       << "\n" << "   dense:  " << dense_filter[index].scaled_state_error();
  // }

  // //===========================================================================
  // // See how the multivariate results stack up vs an equivalent scalar model.
  // TEST_F(MultivariateStateSpaceRegressionModelTest, ScalarComparisonTest) {
  //   int xdim = 3;
  //   int nseries = 1;
  //   int nfactors = 1;
  //   int sample_size = 100;
  //   int test_size = 2;
  //   double residual_sd = .1;

  //   McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                         test_size, residual_sd);
  //   sim.regression_coefficients(0, 1) = 100.0;
  //   sim.build(residual_sd);

  //   // ---------------------------------------------------------------------------
  //   // Build the scalar model
  //   // ---------------------------------------------------------------------------
  //   NEW(StateSpaceRegressionModel, scalar_model)(xdim);
  //   for (int i = 0; i < sim.response.nrow(); ++i) {
  //     NEW(RegressionData, scalar_dp)(sim.response(i, 0), sim.predictors.row(i));
  //     scalar_model->add_regression_data(scalar_dp);
  //   }

  //   scalar_model->observation_model()->set_Beta(sim.regression_coefficients.row(0));
  //   scalar_model->observation_model()->set_sigsq(square(residual_sd));

  //   // TODO: multiply factor_sd by the first observation coefficient.
  //   NEW(LocalLevelStateModel, scalar_trend)(1.0);
  //   // set initial state distribution.
  //   scalar_model->add_state(scalar_trend);

  //   // ---------------------------------------------------------------------------
  //   // Build the multivariate model
  //   // ---------------------------------------------------------------------------
  //   sim.model->observation_model()->clear_methods();
  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
  //   }

  //   set_observation_coefficients(sim.observation_coefficients,
  //                                *sim.state_model);
  //   sim.state_model->clear_methods();

  //   int niter = 100;

  //   Matrix scalar_state_draws(niter, sample_size);
  //   Matrix mv_state_draws(niter, sample_size);
  //   for (int i = 0; i < niter; ++i) {
  //     scalar_model->impute_state(GlobalRng::rng);
  //     // sim.model->impute_state(GlobalRng::rng);
  //     sim.model->sample_posterior();
  //     scalar_state_draws.row(i) = ConstVectorView(
  //         scalar_model->state_contribution(0), 0, sample_size);
  //     mv_state_draws.row(i) = ConstVectorView(
  //         sim.model->state_contributions(0).row(0),
  //         0, sample_size);
  //   }

  //   std::ofstream scalar_draws_file("scalar_state_draws.out");
  //   scalar_draws_file << ConstVectorView(sim.state.row(0), 0, sample_size) << "\n"
  //                     << scalar_state_draws;

  //   std::ofstream mv_state_draws_file("mv_state_draws.out");
  //   mv_state_draws_file << ConstVectorView(sim.state.row(0), 0, sample_size)
  //                       << "\n" << mv_state_draws;
  // }

  // //===========================================================================
  // // Test the full MCMC experience.
  // TEST_F(MultivariateStateSpaceRegressionModelTest, McmcTest) {
  //   // Simulate fake data from the model: shared local level and a regression
  //   // effect.
  //   int xdim = 3;
  //   int nseries = 10;
  //   int nfactors = 1;
  //   int sample_size = 100;
  //   int test_size = 20;
  //   double residual_sd = .1;

  //   McmcTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                         test_size, residual_sd);
  //   sim.regression_coefficients(0, 1) = 100.0;
  //   sim.build(residual_sd);

  //   EXPECT_EQ(sim.xdim(), xdim);
  //   EXPECT_EQ(sim.nseries(), nseries);
  //   EXPECT_EQ(sim.regression_coefficients.nrow(), nseries);
  //   EXPECT_EQ(sim.regression_coefficients.ncol(), xdim);

  //   ofstream("mcmc_raw_data.out") << sim.response;
  //   ofstream("mcmc_predictors.out") << sim.predictors;
  //   ofstream("true_state_contributions.out") << sim.observation_coefficients * sim.state;
  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   // Set parameters to their true values so we can isolate the MCMC to find
  //   // the bug.

  //   //    sim.model->observation_model()->clear_methods();
  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(residual_sd * residual_sd);
  //   }

  //   set_observation_coefficients(sim.observation_coefficients,
  //                                *sim.state_model);
  //   //    sim.state_model->clear_methods();

  //   // sim.model->permanently_set_state(SubMatrix(
  //   //     sim.state, 0, nfactors - 1, 0, sample_size - 1).to_matrix());

  //   // sim.state_model->clear_methods();
  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //   // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  //   int niter = 1000;
  //   int burn = 300;

  //   //---------------------------------------------------------------------------
  //   // Create space to store various different kinds of MCMC draws.
  //   //---------------------------------------------------------------------------

  //   // Draws of the raw state variables.  These might not be identified.
  //   std::vector<Matrix> factor_draws;
  //   for (int i = 0; i < nfactors; ++i) {
  //     Matrix draws(niter, sample_size);
  //     factor_draws.push_back(draws);
  //   }

  //   // The state contribution is the observation coefficients times the factor
  //   // values.

  //   // stored on a series by series basis.
  //   std::vector<Matrix> state_contribution_draws(
  //       nseries, Matrix(niter, sample_size));

  //   // Store the output of calls to predict().
  //   Array prediction_draws(std::vector<int>{
  //       niter, sim.model->nseries(), test_size});

  //   // Store the observation coefficients.
  //   Array observation_coefficient_draws(
  //       std::vector<int>{niter, nseries, nfactors});

  //   Array regression_coefficient_draws(
  //       std::vector<int>{niter, nseries, xdim});

  //   // Draws of the residual standard deviation parameters.
  //   Matrix residual_sd_draws(niter, nseries);

  //   int state_error_dimension = nfactors;
  //   Matrix innovation_sd_draws(niter, state_error_dimension);

  //   // ---------------------------------------------------------------------------
  //   // Workspace for dealing with predictions.
  //   // ---------------------------------------------------------------------------

  //   // Predictor variables for the test set.
  //   Matrix test_predictors;
  //   if (test_size > 0) {
  //     test_predictors = ConstSubMatrix(
  //         sim.predictors,
  //         sample_size, sample_size + test_size - 1,
  //         0, ncol(sim.predictors) - 1).to_matrix();
  //     test_predictors = repeat_rows(test_predictors, nseries);
  //   }

  //   ofstream prediction_out("prediction.draws");
  //   prediction_out << ConstSubMatrix(sim.response, sample_size, sample_size + test_size - 1,
  //                                    0, nseries - 1).transpose();

  //   //---------------------------------------------------------------------------
  //   // Run the MCMC -- burn-in
  //   //---------------------------------------------------------------------------
  //   Selector fully_observed(test_size, true);
  //   sim.model->observe_time_dimension(sample_size + test_size);
  //   EXPECT_EQ(sim.model->time_dimension(), sample_size);
  //   for (int i = 0; i < burn; ++i) {
  //     sim.model->sample_posterior();
  //   }

  //   //---------------------------------------------------------------------------
  //   // Run the MCMC -- main algorithm
  //   //---------------------------------------------------------------------------
  //   for (int i = 0; i < niter; ++i) {
  //     if (i % (100) == 0) {
  //       cout << "------ draw " << i << " of " << niter << " ---------\n";
  //     }
  //     sim.model->sample_posterior();

  //     Matrix local_state_contribution_draw = sim.model->state_contributions(0);
  //     for (int series = 0; series < nseries; ++series) {
  //       state_contribution_draws[series].row(i) =
  //           local_state_contribution_draw.row(series);
  //     }

  //     for (int factor = 0; factor < nfactors; ++factor) {
  //       factor_draws[factor].row(i) = sim.model->shared_state().row(factor);
  //     }
  //     Matrix Z = sim.model->observation_coefficients(0, fully_observed)->dense();
  //     observation_coefficient_draws.slice(i, -1, -1) = Z;
  //     for (int series = 0; series < nseries; ++series) {
  //       residual_sd_draws(i, series) =
  //           sim.model->observation_model()->model(series)->sigma();
  //       regression_coefficient_draws.slice(i, series, -1) =
  //           sim.model->observation_model()->model(series)->Beta();
  //     }

  //     innovation_sd_draws.row(i) =
  //         sqrt(sim.model->state_error_variance(2)->dense().diag());

  //     if (test_size > 0) {
  //     prediction_draws.slice(i, -1, -1) = sim.model->simulate_forecast(
  //         GlobalRng::rng,
  //         test_predictors,
  //         sim.model->shared_state().last_col(),
  //         gather_state_specific_final_state(*sim.model));
  //     prediction_out << prediction_draws.slice(i, -1, -1).to_matrix();
  //     }
  //   }

  //   Vector residual_sd_vector(nseries, residual_sd);
  //   ofstream("residual_sd.draws") << residual_sd_vector << "\n"
  //                                 << residual_sd_draws;
  //   auto status = CheckMcmcMatrix(residual_sd_draws, residual_sd_vector);
  //   EXPECT_TRUE(status.ok) << "Problem with residual sd draws." << status;

  //   // Factor draws are not identified.  Factors * observation coefficients is
  //   // identified.
  //   //
  //   // This section prints out the (maybe unidentified) factor and observation
  //   // coefficient levels.
  //   for (int factor = 0; factor < nfactors; ++factor) {
  //     ConstVectorView true_factor(sim.state.row(factor), 0, sample_size);
  //     status = CheckMcmcMatrix(factor_draws[factor], true_factor);
  //     std::ostringstream fname;
  //     fname << "factor_" << factor << "_draws.out";
  //     ofstream factor_draws_out(fname.str());
  //     factor_draws_out << true_factor << "\n" << factor_draws[factor];

  //     std::ostringstream observation_coefficient_fname;
  //     observation_coefficient_fname
  //         << "observation_coefficient_draws_factor_" << factor;
  //     std::ofstream obs_coef_out(observation_coefficient_fname.str());
  //     obs_coef_out << sim.observation_coefficients.col(0) << "\n"
  //                  << observation_coefficient_draws.slice(-1, -1, 0);

  //     //      EXPECT_TRUE(status.ok) << "Error in factor " << factor << ".  " << status;
  //   }

  //   // Print out the state contribution to each series.  These should be
  //   // identified.
  //   //
  //   // Also print the regression coefficients for each series.

  //   Matrix true_state_contributions = sim.observation_coefficients * sim.state;
  //   for (int series = 0; series < nseries; ++series) {
  //     std::ostringstream fname;
  //     fname << "state_contribution_series_" << series;
  //     std::ofstream state_contribution_out(fname.str());
  //     ConstVectorView truth(true_state_contributions.row(series),
  //                           0, sample_size);
  //     state_contribution_out << truth << "\n"
  //                            << state_contribution_draws[series];
  //     double confidence = .99;
  //     double sd_ratio_threshold = 0.1;
  //     double coverage_fraction = .25;
  //     std::string error_message = CheckStochasticProcess(
  //         state_contribution_draws[series], truth, confidence,
  //         sd_ratio_threshold, coverage_fraction);
  //     EXPECT_EQ(error_message, "") << "Error in state contribution draws for series "
  //                            << series << ".  " << "\n"
  //                            << "See the draws in file " << fname.str()
  //                            << "\n\n";

  //     std::ostringstream reg_fname;
  //     reg_fname << "regression_coefficient_mcmc_draws_series_" << series;
  //     std::ofstream reg_out(reg_fname.str());
  //     reg_out << sim.regression_coefficients.row(series) << "\n"
  //             << regression_coefficient_draws.slice(-1, series, -1);
  //   }

  //   std::ofstream("state_error_sd_mcmc_draws.out")
  //       << Vector(nfactors, 1.0) << "\n" << innovation_sd_draws;
  //   std::cerr << "Low confidence in tests.  Some commented out.\n";
  // }

}  // namespace

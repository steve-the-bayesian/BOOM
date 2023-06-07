#include "gtest/gtest.h"

#include "test_utils/test_utils.hpp"

#include "Models/StateSpace/Multivariate/tests/student_regression_framework.hpp"
#include "Models/StateSpace/Multivariate/tests/student_mcmc_storage.hpp"

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

template <class OBJECT>
void save_to_file(const std::string &filename, const OBJECT &object) {
  std::ofstream out(filename);
  out << object;
}

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
  // Repeat each row of 'mat' a specified number of 'times'.
  //
  // For example.  If a matrix has two rows: A = (r1, r2) then repeat_rows(A, 3)
  // = (r1, r1, r1, r2, r2, r2).
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
  // Set the observation coefficients in a model to the specified set of
  // coefficients.
  void set_observation_coefficients(
      const Matrix &coefficients,
      ConditionallyIndependentSharedLocalLevelStateModel &model) {
    for (int i = 0; i < coefficients.nrow(); ++i) {
      model.raw_observation_coefficients(i)->set_Beta(coefficients.row(i));
    }
  }

  //===========================================================================
  // Check that the framework produces what we expect.
  // TEST_F(StudentMvssRegressionTest, FrameworkTest) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 200;
  //   int test_size = 20;
  //   double residual_sd = .1;
  //   double tail_thickness = 3.0;
  //   StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                            test_size, residual_sd, tail_thickness);

  // }

  // //===========================================================================
  // TEST_F(StudentMvssRegressionTest, EmptyTest) {}

  // //===========================================================================
  // TEST_F(StudentMvssRegressionTest, ConstructorTest) {
  //   StudentMvssRegressionModel model(3, 4);
  // }

  // //===========================================================================
  // TEST_F(StudentMvssRegressionTest, DataTest) {
  //   StudentMultivariateTimeSeriesRegressionData data_point(3.2, Vector{1, 2, 3}, 0, 4);
  //   EXPECT_DOUBLE_EQ(3.2, data_point.y());
  //   EXPECT_TRUE(VectorEquals(Vector{1, 2, 3}, data_point.x()));
  //   EXPECT_EQ(0, data_point.series());
  //   EXPECT_EQ(4, data_point.timestamp());

  //   EXPECT_DOUBLE_EQ(1.0, data_point.weight());
  //   data_point.set_weight(1.2);
  //   EXPECT_DOUBLE_EQ(1.2, data_point.weight());

  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 200;
  //   int test_size = 20;
  //   double residual_sd = .1;
  //   double tail_thickness = 3.0;
  //   StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                            test_size, residual_sd, tail_thickness);

  //   EXPECT_EQ(sim.model->time_dimension(), sample_size);
  //   EXPECT_DOUBLE_EQ(sim.model->observed_data(1, 4),
  //                    sim.response(4, 1));
  // }

  // //===========================================================================
  // // Check if the state can be recovered if there is only one factor and (b) the
  // // model parameters are known.  This is a precursor to MCMC success.
  // TEST_F(StudentMvssRegressionTest, DrawStateTest) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 200;
  //   int test_size = 20;
  //   double residual_sd = .1;
  //   double tail_thickness = 3.0;
  //   StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                            test_size, residual_sd, tail_thickness);

  //   sim.model->observation_model()->clear_methods();

  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(
  //         sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(
  //         residual_sd * residual_sd);
  //     sim.model->observation_model()->model(i)->set_nu(
  //         tail_thickness);
  //   }

  //   sim.model->state_model(0)->clear_methods();
  //   set_observation_coefficients(
  //       sim.observation_coefficients, *sim.trend_model);

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
  //Check draws of the state parameters given observation model parameters, with
  //the state fixed at true values.  This is for the model with just a shared
  //local level.
  // TEST_F(StudentMvssRegressionTest, CheckStateParameterDraws) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 200;
  //   int test_size = 20;
  //   double residual_sd = .1;
  //   double tail_thickness = 3.0;

  //   StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                            test_size, residual_sd, tail_thickness);
  //   int niter = 100;
  //   int burn = 10;

  //   // The observation model parameters are fixed at the true values.
  //   sim.model->observation_model()->clear_methods();

  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(
  //         sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(
  //         residual_sd * residual_sd);
  //     sim.model->observation_model()->model(i)->set_nu(
  //         tail_thickness);
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
  //           sim.trend_model->raw_observation_coefficients(j)->Beta()[0];
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

  //===========================================================================
  // Check draws of the state parameters given observation model parameters,
  // with the state fixed at true values.  This is for the model with both a
  // trend and seasonal components.
  // TEST_F(StudentMvssRegressionTest, CheckSeasonalStateParameterDraws) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 2;
  //   int sample_size = 200;
  //   int test_size = 20;
  //   double residual_sd = .1;
  //   double tail_thickness = 3.0;
  //   int nseasons = 4;

  //   StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                            test_size, residual_sd, tail_thickness,
  //                            nseasons);
  //   int niter = 100;
  //   int burn = 10;
  //   std::string prefix = "SeasonalStateParameterDraws.";

  //   // The observation model parameters are fixed at the true values.
  //   sim.model->observation_model()->clear_methods();
  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(
  //         sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(
  //         residual_sd * residual_sd);
  //     sim.model->observation_model()->model(i)->set_nu(
  //         tail_thickness);
  //   }

  //   sim.model->permanently_set_state(SubMatrix(
  //       sim.state, 0, sim.state_dimension() - 1, 0, sample_size - 1).to_matrix());

  //   Array seasonal_observation_coefficient_draws(std::vector<int>{
  //       niter,
  //       nseries,
  //       nfactors * (nseasons - 1)});

  //   for (int i = 0; i < burn; ++i) {
  //     sim.model->sample_posterior();
  //   }
  //   Selector fully_observed(nseries, true);
  //   for (int i = 0; i < niter; ++i) {
  //     sim.model->sample_posterior();
  //     seasonal_observation_coefficient_draws.slice(i, -1, -1) =
  //         sim.seasonal_model->observation_coefficients(
  //             1, fully_observed)->dense();
  //   }

  //   for (int i = 0; i < nseries; ++i) {
  //     Matrix coefficients = seasonal_observation_coefficient_draws.slice(-1, i, -1).to_matrix();
  //     auto status = CheckMcmcMatrix(
  //         coefficients,
  //         sim.seasonal_observation_coefficients.row(i));
  //     std::ostringstream fname;
  //     fname << prefix << "observation_coefficients_series_" << i;
  //     std::ofstream out(fname.str());
  //     out << sim.seasonal_observation_coefficients.row(i) << "\n"
  //         << coefficients;
  //     EXPECT_TRUE(status.ok) << status;
  //   }
  // }

  // //===========================================================================
  // // Check draws of the regression model parameters given the state model
  // // parameters, with the state fixed at true values.
  // TEST_F(StudentMvssRegressionTest, CheckRegressionParameterDraws) {
  //   int xdim = 3;
  //   int nseries = 2;
  //   int nfactors = 1;
  //   int sample_size = 500;
  //   int test_size = 2;
  //   double residual_sd = .1;
  //   double tail_thickness = 3.0;

  //   StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                            test_size, residual_sd, tail_thickness);
  //   int niter = 200;
  //   int burn = 10;

  //   // The state model parameters are set at the true values.
  //   sim.trend_model->clear_methods();
  //   set_observation_coefficients(sim.observation_coefficients,
  //                                *sim.trend_model);

  //   // The state is permanently set at the true value.
  //   sim.model->permanently_set_state(SubMatrix(
  //       sim.state, 0, nfactors - 1, 0, sample_size - 1).to_matrix());

  //   // Create a bunch of space to store the draws of the model parameters.
  //   std::vector<Matrix> regression_coefficient_draws;
  //   for (int series = 0; series < nseries; ++series) {
  //     Matrix draws(niter, xdim);
  //     regression_coefficient_draws.push_back(draws);
  //   }
  //   Matrix residual_sd_draws(niter, nseries);
  //   Matrix residual_tail_thickness_draws(niter, nseries);

  //   // Run some burn-in iterations.  After this the model should be "converged"
  //   // or close to it.
  //   for (int i = 0; i < burn; ++i) {
  //     sim.model->sample_posterior();
  //   }

  //   // Run 'niter' MCMC steps, storing the draws each time.
  //   for (int i = 0; i < niter; ++i) {
  //     sim.model->sample_posterior();
  //     for (int series = 0; series < nseries; ++series) {
  //       regression_coefficient_draws[series].row(i) =
  //           sim.model->observation_model()->model(series)->Beta();
  //       residual_sd_draws(i, series) =
  //           sim.model->observation_model()->model(series)->sigma();
  //       residual_tail_thickness_draws(i, series) =
  //           sim.model->observation_model()->model(series)->nu();
  //     }
  //   }

  //   // Check the status of the residual SD parameters.  There's one such
  //   // parameter for each series.  We store the draws in a file so they can be
  //   // visually inspected (e.g. using R or python).
  //   auto status = CheckMcmcMatrix(residual_sd_draws, Vector(nseries, residual_sd), 0.99);
  //   EXPECT_TRUE(status.ok) << "Residual SD did not cover true values " << status;
  //   std::ofstream rsd_draws("residual_sd_full_conditional.out");
  //   rsd_draws << Vector(nseries, residual_sd) << "\n" << residual_sd_draws;

  //   // Check the status of the tail thickness ("nu") parameters.  There's one
  //   // such parameter for each series.
  //   status = CheckMcmcMatrix(residual_tail_thickness_draws,
  //                            Vector(nseries, tail_thickness),
  //                            0.99);
  //   EXPECT_TRUE(status.ok)
  //       << "Tail thickness parameter did not cover true values "<< status;
  //   std::ofstream nu_draws("residual_tail_thickness_draws.out");
  //   nu_draws << Vector(nseries, tail_thickness) << "\n"
  //            << residual_tail_thickness_draws;

  //   // Check the status of the regression coefficients.  There is a vector of
  //   // regression coefficients for each series, so we have to do the checks one
  //   // series at a time.  If we run this test with large numbers of series then
  //   // we'll need to do some sort of Bonferroni correction for multiple
  //   // comparisons.
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

  // //===========================================================================
  // // Test the full MCMC experience.
  // TEST_F(StudentMvssRegressionTest, TrendMcmcTest) {
  //   // Simulate fake data from the model: shared local level and a regression
  //   // effect.
  //   int xdim = 3;
  //   int nseries = 5;
  //   int nfactors = 1;
  //   int sample_size = 150;
  //   int test_size = 20;
  //   double residual_sd = .1;
  //   double tail_thickness = 3.0;
  //   int niter = 1000;
  //   int burn = 100;
  //   std::string prefix = "TrendMcmcTest.";

  //   StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
  //                            test_size, residual_sd, tail_thickness);
  //   sim.regression_coefficients(0, 1) = 100.0;
  //   sim.build(residual_sd, tail_thickness);

  //   EXPECT_EQ(sim.xdim(), xdim);
  //   EXPECT_EQ(sim.nseries(), nseries);
  //   EXPECT_EQ(sim.regression_coefficients.nrow(), nseries);
  //   EXPECT_EQ(sim.regression_coefficients.ncol(), xdim);

  //   ofstream(prefix + "raw_data.out") << sim.response;
  //   ofstream(prefix + "predictors.out") << sim.predictors;
  //   ofstream(prefix + "true_state_contributions.out") <<
  //       sim.observation_coefficients * sim.state;

  //   for (int i = 0; i < nseries; ++i) {
  //     sim.model->observation_model()->model(i)->set_Beta(
  //         sim.regression_coefficients.row(i));
  //     sim.model->observation_model()->model(i)->set_sigsq(
  //         residual_sd * residual_sd);
  //     sim.model->observation_model()->model(i)->set_nu(
  //         tail_thickness);
  //   }
  //   set_observation_coefficients(
  //       sim.trend_observation_coefficients,
  //       *sim.trend_model);

  //   //---------------------------------------------------------------------------
  //   // Create space to store various different kinds of MCMC draws.
  //   //---------------------------------------------------------------------------
  //   StudentRegressionMcmcStorage storage(prefix);
  //   int nseasons = -1;
  //   storage.allocate(niter,
  //                    sim.model->nseries(),
  //                    sim.model->xdim(),
  //                    sample_size,
  //                    test_size,
  //                    nfactors,
  //                    nseasons);

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

  //   //---------------------------------------------------------------------------
  //   // Run the MCMC -- burn-in
  //   //---------------------------------------------------------------------------
  //   sim.model->observe_time_dimension(sample_size + test_size);
  //   EXPECT_EQ(sim.model->time_dimension(), sample_size);
  //   cout << "MCMC burn-in...\n";
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
  //     storage.store_mcmc(sim.model.get(), i);
  //     storage.store_prediction(
  //         sim.model.get(), i, test_predictors, GlobalRng::rng);
  //   }
  //   storage.test_residual_sd(Vector(nseries, residual_sd));
  //   storage.test_tail_thickness(Vector(nseries, tail_thickness));
  //   storage.test_factors_and_state_contributions(sim);
  //   storage.test_true_state_contributions(sim);
  //   storage.print_trend_innovation_sd_draws();
  //   storage.print_prediction_draws(sim);
  //   storage.print_regression_coefficients(sim);
  // }

  //===========================================================================
  // Test the full MCMC experience with both a local linear trend and a seasonal
  // component.
  TEST_F(StudentMvssRegressionTest, TrendSeasonalMcmcTest) {
    // Simulate fake data from the model: shared local level, seasonal, and a
    // regression effect.
    int xdim = 3;
    int nseries = 3;
    int nfactors = 1;
    int sample_size = 500;
    int test_size = 20;
    double residual_sd = 10;
    double tail_thickness = 3.0;
    int niter = 1000;
    int burn = 100;
    int nseasons = 4;
    std::string prefix = "TrendSeasonalMcmcTest.";

    StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
                             test_size, residual_sd, tail_thickness,
                             nseasons);
    sim.regression_coefficients(0, 1) = 100.0;
    sim.seasonal_observation_coefficients(1, 0) *= 100;
    sim.combine_observation_coefficients();
    sim.build(residual_sd, tail_thickness);

    // If we freeze the model parameters at true values we recover the state.
    //
    // sim.freeze_seasonal_parameters();
    // sim.freeze_trend_parameters();
    // sim.freeze_observation_model_parameters();
    // sim.set_model_parameters_to_true_values();

    // If we freeze the state at its true values do we recover the model
    // parameters?
    // sim.freeze_state_at_truth();

    std::cout << "sim.seasonal_observation_coefficients: \n"
              << sim.seasonal_observation_coefficients;

    EXPECT_EQ(sim.xdim(), xdim);
    EXPECT_EQ(sim.nseries(), nseries);
    EXPECT_EQ(sim.regression_coefficients.nrow(), nseries);
    EXPECT_EQ(sim.regression_coefficients.ncol(), xdim);

    ofstream(prefix + "raw_data.out") << sim.response;
    ofstream(prefix + "predictors.out") << sim.predictors;
    ofstream(prefix + "true_state_contributions.out") <<
        sim.observation_coefficients * sim.state;

    sim.set_model_parameters_to_true_values();

    //---------------------------------------------------------------------------
    // Create space to store various different kinds of MCMC draws.
    //---------------------------------------------------------------------------
    StudentRegressionMcmcStorage storage(prefix);
    storage.allocate(niter,
                     sim.model->nseries(),
                     sim.model->xdim(),
                     sample_size,
                     test_size,
                     nfactors,
                     nseasons);

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

    //---------------------------------------------------------------------------
    // Run the MCMC -- burn-in
    //---------------------------------------------------------------------------
    sim.model->observe_time_dimension(sample_size + test_size);
    EXPECT_EQ(sim.model->time_dimension(), sample_size);
    cout << "MCMC burn-in...\n";
    for (int i = 0; i < burn; ++i) {
      //      std::cout << "burn in iteration " << i << "\n";
      sim.model->sample_posterior();
    }

    //---------------------------------------------------------------------------
    // Run the MCMC -- main algorithm
    //---------------------------------------------------------------------------
    for (int i = 0; i < niter; ++i) {
      if (i % (100) == 0) {
        cout << "======= draw " << i << " of " << niter << " =========\n";
      }
      sim.model->sample_posterior();
      storage.store_mcmc(sim.model.get(),
                         sim.trend_model.get(), sim.
                         seasonal_model.get(),
                         i);
      storage.store_prediction(
          sim.model.get(), i, test_predictors, GlobalRng::rng);
    }
    storage.test_residual_sd(Vector(nseries, residual_sd));
    storage.test_tail_thickness(Vector(nseries, tail_thickness));
    storage.test_factors_and_state_contributions(sim);
    storage.test_true_state_contributions(sim);
    storage.print_trend_draws(sim);
    storage.print_seasonal_draws(sim);
    storage.print_trend_innovation_sd_draws();
    storage.print_prediction_draws(sim);
    storage.print_regression_coefficients(sim);
    storage.print_observation_coefficients(sim);
  }


}  // namespace

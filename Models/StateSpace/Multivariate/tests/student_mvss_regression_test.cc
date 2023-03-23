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

  void reorder_state_contribution(
      const Matrix &input,
      std::vector<Matrix> &output,
      int iteration) {
    int nseries = input.nrow();
    for (int series = 0; series < nseries; ++series) {
      output[series].row(iteration) = input.row(series);
    }
  }

  //===========================================================================
  template <class MULTIVARIATE_MODEL>
  std::vector<Vector> gather_state_specific_final_state(
      const MULTIVARIATE_MODEL &model) {
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
  class McmcStorage {
   public:

    // Args:
    //   prefix: A string identifying the test case being run.  This will be
    //     prepended to the names of the output files to keep output from one
    //     test case from overwriting another.
    McmcStorage(const std::string &prefix)
        : prefix_(prefix)
    {}

    // Allocate space to store a specifiec number of iterations.
    //
    // Args:
    //   niter:  The number of MCMC iterations to allocate.
    //   nseries:  The number of time series in the model.
    //   xdim:  The number of predictor variables in the regression.
    //   sample_size:  The number of time points in the training data.
    //   test_size:  The number of time points in the holdout data.
    //   nfactors:  The number of factors in the local level state model.
    //   nseasons: The number of seasons in the seasonal component of the model,
    //     if there is one.  If nseasons < 2 then no space for seasonal output
    //     will be allocated.
    void allocate(int niter,
                  int nseries,
                  int xdim,
                  int sample_size,
                  int test_size,
                  int nfactors,
                  int nseasons) {
      factor_draws_ = std::vector<Matrix>(nfactors, Matrix(
          niter, sample_size));
      trend_contribution_draws_ = std::vector<Matrix>(
          nseries, Matrix(niter, sample_size));
      if (nseasons > 1) {
        seasonal_contribution_draws_ = std::vector<Matrix>(
            nseries, Matrix(niter, sample_size));
      }

      prediction_draws_ = Array(std::vector<int>{
          niter, nseries, test_size});

      int trend_dim = nfactors;
      int seasonal_dim = 0;
      if (nseasons > 1) {
        seasonal_dim = nfactors * (nseasons - 1);
      }
      int state_dim = trend_dim + seasonal_dim;

      observation_coefficient_draws_ = Array(
          std::vector<int>{niter, nseries, state_dim});
      regression_coefficient_draws_ = Array(
          std::vector<int>{niter, nseries, xdim});
      residual_sd_draws_ = Matrix(niter, nseries);
      tail_thickness_draws_ = Matrix(niter, nseries);
      trend_innovation_sd_draws_ = Matrix(niter, nfactors);
      fully_observed_ = Selector(nseries, true);
    }

    //---------------------------------------------------------------------------
    // The number of factors in the local level state model.
    int nfactors() const {
      return factor_draws_.size();
    }

    //---------------------------------------------------------------------------
    // The number of time series in the model.
    int nseries() const {
      return fully_observed_.nvars_possible();
    }

    //---------------------------------------------------------------------------
    // The number of time points in the training data.
    int sample_size() const {
      if (factor_draws_.empty()) {
        return 0;
      } else {
        return factor_draws_[0].ncol();
      }
    }

    //---------------------------------------------------------------------------
    // The number of time points in the holdout data.
    int test_size() const {
      return prediction_draws_.empty() ? 0 : prediction_draws_.dim(2);
    }

    //---------------------------------------------------------------------------
    // Store the results of the most recent MCMC draw.
    //
    // Args:
    //   model:  The model containing the values to be stored.
    //   iteration:  The iteration number being stored.
    void store_mcmc(
        const StudentMvssRegressionModel *model,
        const ConditionallyIndependentSharedLocalLevelStateModel *trend_model,
        const SharedSeasonalStateModel *seasonal_model,
        int iteration) {
      reorder_state_contribution(model->state_contributions(0),
                                 trend_contribution_draws_,
                                 iteration);
      if (model->number_of_state_models() > 1) {
        reorder_state_contribution(model->state_contributions(1),
                                   seasonal_contribution_draws_,
                                   iteration);
      }

      for (int factor = 0; factor < nfactors(); ++factor) {
        factor_draws_[factor].row(iteration) = model->shared_state().row(factor);
      }

      Matrix Z = model->observation_coefficients(0, fully_observed_)->dense();
      observation_coefficient_draws_.slice(iteration, -1, -1) = Z;
      for (int series = 0; series < model->nseries(); ++series) {
        residual_sd_draws_(iteration, series) =
            model->observation_model()->model(series)->sigma();
        tail_thickness_draws_(iteration, series) =
            model->observation_model()->model(series)->nu();
        regression_coefficient_draws_.slice(iteration, series, -1) =
            model->observation_model()->model(series)->Beta();
      }

      for (int factor = 0; factor < nfactors(); ++factor) {
        trend_innovation_sd_draws_(iteration, factor) =
            trend_model->innovation_model(factor)->sd();
      }
    }

    //---------------------------------------------------------------------------
    // Store a draw from the posterior predictive distribution.
    // Args:
    //   model:  The model making the prediction.
    //   iteration:  The iteration number indexing the draw to be stored.
    //   test_predictors:  The X variables where the prediction is to be made.
    //   rng: A random number generator used to drive the draw from the
    //     posterior predictive distribution.
    void store_prediction(StudentMvssRegressionModel *model,
                          int iteration,
                          const Matrix test_predictors,
                          RNG &rng) {
        prediction_draws_.slice(iteration, -1, -1) = model->simulate_forecast(
            rng,
            test_predictors,
            model->shared_state().last_col(),
            gather_state_specific_final_state(*model));
    }

    //---------------------------------------------------------------------------
    void test_residual_sd(const Vector &residual_sd_vector) {
      std::ofstream(prefix_ + "residual_sd.draws") << residual_sd_vector << "\n"
                                                   << residual_sd_draws_;
      double confidence = .99;
      bool control_multiple_comparisons = true;
      auto status = CheckMcmcMatrix(residual_sd_draws_, residual_sd_vector,
                                    confidence, control_multiple_comparisons);
      EXPECT_TRUE(status.ok) << "Problem with residual sd draws." << status;
    }

    //---------------------------------------------------------------------------
    void test_tail_thickness(const Vector &tail_thickness_vector) {
      ofstream(prefix_ + "tail_thickness.draws") << tail_thickness_vector << "\n"
                                                 << tail_thickness_draws_;
      auto status = CheckMcmcMatrix(tail_thickness_draws_, tail_thickness_vector);
      EXPECT_TRUE(status.ok) << "Problem with tail thickness draws." << status;
    }

    //---------------------------------------------------------------------------
    void test_factors_and_state_contributions(const StudentTestFramework &sim) {
      // Factor draws are not identified.  Factors * observation coefficients is
      // identified.
      //
      // This section prints out the (maybe unidentified) factor and observation
      // coefficient levels.
      for (int factor = 0; factor < nfactors(); ++factor) {
        ConstVectorView true_factor(sim.state.row(factor), 0, sample_size());
        std::ostringstream fname;
        fname << prefix_ + "factor_" << factor << "_draws.out";
        ofstream factor_draws_out(fname.str());
        factor_draws_out << true_factor << "\n" << factor_draws_[factor];

        std::ostringstream observation_coefficient_fname;
        observation_coefficient_fname
            << prefix_ << "observation_coefficient_draws_factor_" << factor;
        std::ofstream obs_coef_out(observation_coefficient_fname.str());
        obs_coef_out << sim.observation_coefficients.col(0) << "\n"
                     << observation_coefficient_draws_.slice(-1, -1, 0);
      }
    }

    // Check that the state contributions to each time series match the true
    // underlying state.  This is a better test than testing the individual
    // factors.  The factors are not identified, but the sums of the state
    // contributions are identified.
    void test_true_state_contributions(const StudentTestFramework &sim) {
      Matrix true_state_contributions =
          sim.observation_coefficients * sim.state;

      Matrix true_trend_contributions =
          sim.trend_observation_coefficients * sim.trend_state;

      Matrix true_seasonal_contributions;
      if (sim.nseasons() > 1) {
        true_seasonal_contributions =
            sim.seasonal_observation_coefficients * sim.seasonal_state;
      }

      for (int series = 0; series < nseries(); ++series) {
        std::ostringstream fname;
        fname << prefix_ << "trend_contribution_series_" << series;
        std::ofstream trend_contribution_out(fname.str());
        ConstVectorView truth(true_trend_contributions.row(series),
                              0, sample_size());
        trend_contribution_out << truth << "\n"
                               << trend_contribution_draws_[series];
        EXPECT_TRUE(CheckTrend(trend_contribution_draws_[series], truth, .9))
            << "The inferred trend contribution for series " << series
            << " is not closely aligned with the true trend contribution "
            << "values.";

        if (sim.nseasons() > 1) {
          std::ostringstream fname;
          fname << prefix_ << "seasonal_contribution_series_" << series;
          std::ofstream seasonal_contribution_out(fname.str());
          ConstVectorView truth(true_seasonal_contributions.row(series),
                                0, sample_size());
          seasonal_contribution_out << truth << "\n"
                                    << seasonal_contribution_draws_[series];
          EXPECT_TRUE(CheckTrend(seasonal_contribution_draws_[series], truth, .9))
              << "The inferred seasonal contribution for series " << series
              << " is not closely aligned with the true seasonal contribution "
              << "values.";
        }
      }
    }

    //---------------------------------------------------------------------------
    void print_regression_coefficients(const StudentTestFramework &sim) {
      for (int series = 0; series < nseries(); ++series) {
        std::ostringstream reg_fname;
        reg_fname << prefix_
                  << "regression_coefficient_mcmc_draws_series_"
                  << series;
        std::ofstream reg_out(reg_fname.str());
        reg_out << sim.regression_coefficients.row(series) << "\n"
                << regression_coefficient_draws_.slice(-1, series, -1);
      }
    }

    //---------------------------------------------------------------------------
    void print_trend_innovation_sd_draws() {
      std::ofstream(prefix_ + "state_error_sd_mcmc_draws.out")
          << Vector(nfactors(), 1.0) << "\n" << trend_innovation_sd_draws_;
    }

    //---------------------------------------------------------------------------
    void print_prediction_draws(const StudentTestFramework &sim) {
      ofstream prediction_out(prefix_ + "prediction.draws");
      prediction_out << ConstSubMatrix(sim.response,
                                       sample_size(),
                                       sample_size() + test_size() - 1,
                                       0, nseries() - 1).transpose();
    }

   private:
    // Output files will have prefix_ prepended to their name so that output
    // from different tests won't over-write each other.
    std::string prefix_;

    // Draws of the raw state variables.  These might not be identified.
    std::vector<Matrix> factor_draws_;

    // The state contribution is the observation coefficients times the factor
    // values.
    std::vector<Matrix> trend_contribution_draws_;
    std::vector<Matrix> seasonal_contribution_draws_;

    // Store the output of calls to predict().
    Array prediction_draws_;

    Array observation_coefficient_draws_;
    Array trend_observation_coefficient_draws_;
    Array seasonal_observation_coefficient_draws_;

    Array regression_coefficient_draws_;

    Matrix residual_sd_draws_;
    Matrix trend_innovation_sd_draws_;
    Matrix tail_thickness_draws_;

    Selector fully_observed_;
  };

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
  // //===========================================================================
  // // Check draws of the state parameters given observation model parameters,
  // // with the state fixed at true values.
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
  //   McmcStorage storage(prefix);
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
    // Simulate fake data from the model: shared local level and a regression
    // effect.
    int xdim = 3;
    int nseries = 5;
    int nfactors = 1;
    int sample_size = 250;
    int test_size = 20;
    double residual_sd = .1;
    double tail_thickness = 3.0;
    int niter = 1000;
    int burn = 100;
    int nseasons = 4;
    std::string prefix = "TrendSeasonalMcmcTest.";

    StudentTestFramework sim(xdim, nseries, nfactors, sample_size,
                             test_size, residual_sd, tail_thickness,
                             nseasons);
    sim.regression_coefficients(0, 1) = 100.0;
    sim.build(residual_sd, tail_thickness);

    EXPECT_EQ(sim.xdim(), xdim);
    EXPECT_EQ(sim.nseries(), nseries);
    EXPECT_EQ(sim.regression_coefficients.nrow(), nseries);
    EXPECT_EQ(sim.regression_coefficients.ncol(), xdim);

    ofstream(prefix + "raw_data.out") << sim.response;
    ofstream(prefix + "predictors.out") << sim.predictors;
    ofstream(prefix + "true_state_contributions.out") <<
        sim.observation_coefficients * sim.state;

    for (int i = 0; i < nseries; ++i) {
      sim.model->observation_model()->model(i)->set_Beta(
          sim.regression_coefficients.row(i));
      sim.model->observation_model()->model(i)->set_sigsq(
          residual_sd * residual_sd);
      sim.model->observation_model()->model(i)->set_nu(
          tail_thickness);
    }
    set_observation_coefficients(
        sim.trend_observation_coefficients,
        *sim.trend_model);

    //---------------------------------------------------------------------------
    // Create space to store various different kinds of MCMC draws.
    //---------------------------------------------------------------------------
    McmcStorage storage(prefix);
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
    storage.print_trend_innovation_sd_draws();
    storage.print_prediction_draws(sim);
    storage.print_regression_coefficients(sim);
  }


}  // namespace

#ifndef BOOM_STATESPACE_MULTIVARIATE_TESTING_STUDENT_MCMC_STORAGE_HPP_
#define BOOM_STATESPACE_MULTIVARIATE_TESTING_STUDENT_MCMC_STORAGE_HPP_
/*
  Copyright (C) 2005-2023 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "LinAlg/Array.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SubMatrix.hpp"

#include "Models/StateSpace/Multivariate/StudentMvssRegressionModel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedSeasonal.hpp"

#include "Models/StateSpace/Multivariate/tests/student_regression_framework.hpp"

namespace BoomStateSpaceTesting {
  using namespace BOOM;

  // Args:
  //   input: A Matrix containing the state contribution draw for a single state
  //     component (e.g. the trend).  The rows of the Matrix are the series of
  //     the multivariate time series.  The columns are different time points.
  //     The entries are the contributions of the selected state component to
  //     each of the series at each time point.
  //   output: The input values are stored in a different data structure indexed
  //     by Series.  Each vector element is a Matrix giving the MCMC draws for
  //     that time series.
  //   iteration:  The iteration number of the draw being stored.
  inline void reorder_state_contribution(
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
  class StudentRegressionMcmcStorage {
   public:

    // Args:
    //   prefix: A string identifying the test case being run.  This will be
    //     prepended to the names of the output files to keep output from one
    //     test case from overwriting another.
    StudentRegressionMcmcStorage(const std::string &prefix)
        : prefix_(prefix)
    {}

    // Allocate space to store a specific number of iterations.
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
      trend_state_draws_ = Matrix(niter, nfactors * sample_size);
      seasonal_state_draws_ = Matrix(niter, nfactors * sample_size);

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
      trend_observation_coefficient_draws_ = Array(
          std::vector<int>{niter, nseries, nfactors});
      seasonal_observation_coefficient_draws_ = Array(
          std::vector<int>{niter, nseries, seasonal_dim});

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

      // Store the state contributions to each series, by type of state model.
      reorder_state_contribution(model->state_contributions(0),
                                 trend_contribution_draws_,
                                 iteration);
      if (model->number_of_state_models() > 1) {
        reorder_state_contribution(model->state_contributions(1),
                                   seasonal_contribution_draws_,
                                   iteration);
      }

      // Store the factors in the trend model.
      for (int factor = 0; factor < nfactors(); ++factor) {
        factor_draws_[factor].row(iteration) = model->shared_state().row(factor);
      }

      ConstSubMatrix trend(model->shared_state(), 0, nfactors() - 1, 0, sample_size() - 1);
      trend_state_draws_.row(iteration) = Matrix(trend).stack_rows();

      const Selector &current_factors(seasonal_model->current_factors());
      ConstSubMatrix seasonal(model->shared_state(),
                              nfactors(),
                              model->state_dimension() - 1,
                              0,
                              sample_size() - 1);
      Matrix current_seasonal = current_factors.select_rows(seasonal);

      seasonal_state_draws_.row(iteration) = current_seasonal.stack_rows();

      // Store the observation coefficients, overall, and by treand and seasonal.
      Matrix Z = model->observation_coefficients(0, fully_observed_)->dense();
      observation_coefficient_draws_.slice(iteration, -1, -1) = Z;
      Z = trend_model->observation_coefficients(0, fully_observed_)->dense();
      trend_observation_coefficient_draws_.slice(iteration, -1, -1) = Z;
      if (model->number_of_state_models() > 1) {
        Z = seasonal_model->observation_coefficients(0, fully_observed_)->dense();
        seasonal_observation_coefficient_draws_.slice(iteration, -1, -1) = Z;
      }

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

    void print_observation_coefficients(const StudentTestFramework &sim) {
      for (int series = 0; series < nseries(); ++series) {
        std::ostringstream trend_fname;
        trend_fname << prefix_
                    << "trend_observation_coefficients_series_"
                    << series;
        std::ofstream trend_out(trend_fname.str());
        trend_out << rbind(
            sim.trend_observation_coefficients.row(series),
            trend_observation_coefficient_draws_.slice(
                -1, series, -1).to_matrix());

        std::ostringstream seasonal_fname;
        seasonal_fname << prefix_
                       << "seasonal_observation_coefficients_series_"
                       << series;
        std::ofstream seasonal_out(seasonal_fname.str());
        seasonal_out << rbind(
            sim.seasonal_observation_coefficients.row(series),
            seasonal_observation_coefficient_draws_.slice(
                -1, series, -1).to_matrix());
      }
    }

    //---------------------------------------------------------------------------
    // The first row is the true set of trend values.  The subsequent rows are
    // MCMC draws of trend.  If there are multiple factors, then the trend is a
    // matrix with factors as rows and timestamps as columns.  The matrix is
    // flattened to factor0, factor1, factor2, etc.  A single row with nfactors
    // * ntimes entries.
    void print_trend_draws(const StudentTestFramework &sim) {
      ofstream out (prefix_ + "trend_state.draws");
      Matrix true_trend(ConstSubMatrix(
          sim.trend_state, 0, nfactors() - 1, 0, sample_size() - 1));
      out << true_trend.stack_rows() << "\n" << trend_state_draws_;
    }
    //---------------------------------------------------------------------------
    void print_seasonal_draws(const StudentTestFramework &sim) {
      std::vector<Vector> true_seasonal;

      int cursor = 0;
      for (int factor = 0; factor < nfactors(); ++factor) {
        true_seasonal.push_back(
            ConstVectorView(
                sim.seasonal_state.row(cursor), 0, sample_size()));
        cursor += sim.nseasons() - 1;
      }

      ofstream out(prefix_ + "seasonal_state.draws");
      out << concat(true_seasonal) << "\n" << seasonal_state_draws_;
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

    // trend_state_draws_.row(iteration) contains the concatenation of the
    // 'nfactors' values of the current trend effects.  The first 'sample_size'
    // entries are factor 1.  The next 'sample_size' are factor 2, etc.
    Matrix trend_state_draws_;

    // seasonal_state_draws_.row(iteration) contains the concatenation of the
    // 'nfactors' values of the current seasonal effects.  The first
    // 'sample_size' entries are factor 1.  The next 'sample_size' are factor 2,
    // etc.
    Matrix seasonal_state_draws_;

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

}  // namespace BoomStateSpaceTesting
#endif  // BOOM_STATESPACE_MULTIVARIATE_TESTING_STUDENT_MCMC_STORAGE_HPP_

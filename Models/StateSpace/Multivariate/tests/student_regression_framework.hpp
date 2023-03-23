#ifndef BOOM_STATE_SPACE_STUDENT_MULTIVARIATE_TEST_FRAMEWORK_HPP_
#define BOOM_STATE_SPACE_STUDENT_MULTIVARIATE_TEST_FRAMEWORK_HPP_

#include "Models/StateSpace/Multivariate/StudentMvssRegressionModel.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/UniformModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "Models/Glm/PosteriorSamplers/TRegressionSampler.hpp"
#include "Models/Glm/PosteriorSamplers/IndependentRegressionModelsPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedLocalLevel.hpp"
#include "Models/StateSpace/Multivariate/StateModels/SharedSeasonal.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedSeasonalPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/StudentMvssPosteriorSampler.hpp"

#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"

#include "distributions.hpp"

namespace BoomStateSpaceTesting {
  using namespace BOOM;

  // A test framework consisting of fake data simulated from a model, a model
  // object set up to learn the parameters from the fake data, and a set of test
  // data that can be used to measure model forecasts.
  struct StudentTestFramework {
    // Args:
    //   xdim:  The number of predictor variables
    //   nseries:  The number of Y variables.
    //   nfactors:  The number of factors shared by the Y variables.
    //   sample_size: The number of time points to simulate in the training
    //     data.
    //   test_size:  The number of time points to simulate in the test data.
    //   factor_sd: The standard deviation of the innovation terms in the random
    //     walk models describing the state.
    //   residual_sd: The standard deviation of the residuals in the observation
    //     equation.  Each of the 'nseries' responses has the same SD.
    //   df: The degrees of freedom parameter for the residuals in the
    //     observation equation.  Each of the 'nseries' responses has the same
    //     degrees of freedom.
    //   nseasons: The number of seasons in the seasonal component.  If less
    //     than 1 then no seasonal component will be included.
    explicit StudentTestFramework(int xdim,
                                  int nseries,
                                  int nfactors,
                                  int sample_size,
                                  int test_size,
                                  double residual_sd,
                                  double df,
                                  int nseasons = -1)
        : sample_size_(sample_size),
          test_size_(test_size),
          nseasons_(nseasons),
          state(nfactors, sample_size + test_size),
          observation_coefficients(nseries, nfactors),
          regression_coefficients(nseries, xdim),
          predictors(sample_size + test_size, xdim),
          response(sample_size + test_size, nseries),
          model(new StudentMvssRegressionModel(xdim, nseries)),
          trend_model(nullptr),
          seasonal_model(nullptr)
    {
      set_trend_observation_coefficients(nfactors);
      set_seasonal_observation_coefficients(nfactors);
      combine_observation_coefficients();

      // Set up the regression coefficients and the predictors.
      regression_coefficients.randomize();
      predictors.randomize();

      build(residual_sd, df);
    }

    void set_trend_observation_coefficients(int nfactors) {
      trend_observation_coefficients.resize(nseries(), nfactors);
      trend_observation_coefficients.randomize();
      for (int i = 0; i < nfactors; ++i) {
        trend_observation_coefficients(i, i) = 1.0;
        for (int j = i + 1; j < nfactors; ++j) {
          trend_observation_coefficients(i, j) = 0.0;
        }
      }
      trend_observation_coefficients *= 10;
      trend_observation_coefficients.diag() /= 10;
    }

    void set_seasonal_observation_coefficients(int nfactors) {
      if (nseasons_ > 1) {
        int nrow = nseries();
        int ncol = (nseasons_ - 1) * nfactors;
        seasonal_observation_coefficients.resize(nrow, ncol);
        seasonal_observation_coefficients = 0.0;
        for (int i = 0; i < nrow; ++i) {
          for (int j = 0; j < ncol; j += (nseasons_ - 1)) {
            seasonal_observation_coefficients(i, j) = rnorm(0, 1);
          }
        }
      }
    }

    void combine_observation_coefficients() {
      if (nseasons_ > 1) {
        observation_coefficients = cbind(trend_observation_coefficients,
                                         seasonal_observation_coefficients);
      } else {
        observation_coefficients = trend_observation_coefficients;
      }
    }

    //---------------------------------------------------------------------------
    int nfactors() const {
      return trend_observation_coefficients.ncol();
    }

    int nseries() const {
      return regression_coefficients.nrow();
    }

    int nseasons() const {
      return nseasons_;
    }

    int xdim() const {
      return regression_coefficients.ncol();
    }

    void build(double residual_sd, double df) {
      simulate_state();
      simulate_response(residual_sd, df);
      build_model(residual_sd, df);
    }

    // Simulate state from a multivariate local level model.
    void simulate_state() {
      double factor_sd = 1.0;
      trend_state.resize(nfactors(), sample_size_ + test_size_);
      for (int factor = 0; factor < nfactors(); ++factor) {
        trend_state(factor, 0) = rnorm();
        for (int time = 1; time < sample_size_ + test_size_; ++time) {
          trend_state(factor, time) = trend_state(factor, time - 1) + rnorm(0, factor_sd);
        }
      }
      state = trend_state;

      if (nseasons_ > 1) {
        std::vector<Matrix> seasonal_state_components;
        seasonal_state.resize(nfactors() * (nseasons_ - 1), sample_size_ + test_size_);
        for (int factor = 0; factor < nfactors(); ++factor) {
          Vector pattern(nseasons_);
          pattern.randomize();
          seasonal_state_components.push_back(simulate_seasonal_state(
              pattern, factor_sd, sample_size_ + test_size_));
        }
        seasonal_state = rbind(seasonal_state_components);
        state = rbind(trend_state, seasonal_state);
      }

    }

    void simulate_response(double residual_sd, double df) {
      for (int i = 0; i < sample_size_ + test_size_; ++i) {
        Vector yhat = observation_coefficients * state.col(i)
            + regression_coefficients * predictors.row(i);
        for (int j = 0; j < nseries(); ++j) {
          response(i, j) = yhat[j] + rstudent(0, residual_sd, df);
        }
      }
    }

    void add_data_to_model() {
      for (int time = 0; time < sample_size_; ++time) {
        for (int series = 0; series < nseries(); ++series) {
          NEW(StudentMultivariateTimeSeriesRegressionData, data_point)(
              response(time, series), predictors.row(time), series, time);
          model->add_data(data_point);
        }
      }
    }

    void define_state() {
      define_trend();
      define_seasonal();
    }

    void define_trend() {
      trend_model.reset(new ConditionallyIndependentSharedLocalLevelStateModel(
          model.get(), nfactors(), nseries()));
      std::vector<Ptr<GammaModelBase>> innovation_precision_priors;
      for (int factor = 0; factor < nfactors(); ++factor) {
        innovation_precision_priors.push_back(new ChisqModel(1.0, .10));
      }
      Matrix observation_coefficient_prior_mean(nseries(), nfactors(), 0.0);

      NEW(MvnModel, slab)(Vector(nfactors(), 0.0), SpdMatrix(nfactors(), 1.0));
      NEW(VariableSelectionPrior, spike)(nfactors(), 1.0);
      std::vector<Ptr<VariableSelectionPrior>> spikes;
      for (int i = 0; i < nseries(); ++i) {
        spikes.push_back(spike->clone());
      }

      std::vector<Ptr<UnivParams>> sigsq_params;
      for (int i = 0; i < nseries(); ++i) {
        sigsq_params.push_back(model->observation_model()->model(i)->Sigsq_prm());
      }

      NEW(ConditionallyIndependentSharedLocalLevelPosteriorSampler,
          trend_model_sampler)(
              trend_model.get(),
              std::vector<Ptr<MvnBase>>(nseries(), slab),
              spikes,
              sigsq_params);
      trend_model->set_method(trend_model_sampler);
      trend_model->set_initial_state_mean(ConstVectorView(state.col(0), 0, nfactors()));
      trend_model->set_initial_state_variance(SpdMatrix(nfactors(), 1.0));
      model->add_state(trend_model);
    }

    void define_seasonal() {
      if (nseasons_ > 1) {
        seasonal_model.reset(new SharedSeasonalStateModel(
            model.get(), nfactors(), nseasons_, 1));

        Matrix observation_coefficient_prior_mean(nseries(), nfactors(), 0.0);
        NEW(MvnModel, slab)(Vector(nfactors(), 0.0), SpdMatrix(nfactors(), 1.0));
        NEW(VariableSelectionPrior, spike)(nfactors(), 1.0);
        std::vector<Ptr<VariableSelectionPrior>> spikes;
        for (int i = 0; i < nseries(); ++i) {
          spikes.push_back(spike->clone());
        }

        std::vector<Ptr<UnivParams>> sigsq_params;
        for (int i = 0; i < nseries(); ++i) {
          sigsq_params.push_back(model->observation_model()->model(i)->Sigsq_prm());
        }

        NEW(SharedSeasonalPosteriorSampler, seasonal_model_sampler)(
            seasonal_model.get(),
            std::vector<Ptr<MvnBase>>(nseries(), slab),
            spikes,
            sigsq_params,
            GlobalRng::rng);

        seasonal_model->set_method(seasonal_model_sampler);
        int dim = seasonal_model->state_dimension();
        int trend_dim = nfactors();
        seasonal_model->set_initial_state_mean(ConstVectorView(state.col(0), trend_dim));
        seasonal_model->set_initial_state_variance(SpdMatrix(dim, 1.0));
        model->add_state(seasonal_model);
      }
    }

    void set_observation_model_sampler(double residual_sd, double df) {
      for (int i = 0; i < nseries(); ++i) {
        Vector beta_prior_mean(xdim(), 0.0);
        SpdMatrix beta_precision(xdim(), .0001);
        NEW(MvnModel, beta_prior)(beta_prior_mean, beta_precision, true);
        NEW(ChisqModel, residual_precision_prior)(1.0, square(residual_sd));
        NEW(UniformModel, tail_thickness_prior)(1.0, 100.0);
        NEW(CompleteDataStudentRegressionPosteriorSampler, regression_sampler)(
            model->observation_model()->model(i),
            beta_prior,
            residual_precision_prior,
            tail_thickness_prior);
        regression_sampler->set_sigma_upper_limit(100.0);
        model->observation_model()->model(i)->set_method(regression_sampler);
      }
      NEW(IndependentGlmsPosteriorSampler<CompleteDataStudentRegressionModel>,
          observation_model_sampler)(model->observation_model());
      model->observation_model()->set_method(observation_model_sampler);
    }

    void set_posterior_sampler() {
      NEW(StudentMvssPosteriorSampler, sampler)(model.get());
      model->set_method(sampler);
    }

    void build_model(double residual_sd, double df) {
      model.reset(new StudentMvssRegressionModel(xdim(), nseries()));
      add_data_to_model();
      define_state();
      set_observation_model_sampler(residual_sd, df);
      set_posterior_sampler();
    }

    //---------------------------------------------------------------------------
    // Data section
    //---------------------------------------------------------------------------
    int sample_size_;
    int test_size_;

    // The number of seasons in the seasonal component.  A number less than 2 is
    // a signal that there is no seasonal component.
    int nseasons_;

    Matrix state;
    Matrix trend_state;
    Matrix seasonal_state;

    Matrix observation_coefficients;
    Matrix trend_observation_coefficients;
    Matrix seasonal_observation_coefficients;

    Matrix regression_coefficients;
    Matrix predictors;
    Matrix response;

    Ptr<StudentMvssRegressionModel> model;
    Ptr<ConditionallyIndependentSharedLocalLevelStateModel> trend_model;
    Ptr<SharedSeasonalStateModel> seasonal_model;
  };

}  // namespace BoomStateSpaceTesting


#endif  //  BOOM_STATE_SPACE_STUDENT_MULTIVARIATE_TEST_FRAMEWORK_HPP_

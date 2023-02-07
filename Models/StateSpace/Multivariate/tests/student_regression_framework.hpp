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
#include "Models/StateSpace/Multivariate/PosteriorSamplers/SharedLocalLevelPosteriorSampler.hpp"
#include "Models/StateSpace/Multivariate/PosteriorSamplers/StudentMvssPosteriorSampler.hpp"

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
    explicit StudentTestFramework(int xdim,
                                  int nseries,
                                  int nfactors,
                                  int sample_size,
                                  int test_size,
                                  double residual_sd,
                                  double df)
        : sample_size_(sample_size),
          test_size_(test_size),
          state(nfactors, sample_size + test_size),
          observation_coefficients(nseries, nfactors),
          regression_coefficients(nseries, xdim),
          predictors(sample_size + test_size, xdim),
          response(sample_size + test_size, nseries),
          model(new StudentMvssRegressionModel(xdim, nseries))
    {
      observation_coefficients.randomize();
      for (int i = 0; i < nfactors; ++i) {
        observation_coefficients(i, i) = 1.0;
        for (int j = i + 1; j < nfactors; ++j) {
          observation_coefficients(i, j) = 0.0;
        }
      }
      observation_coefficients *= 10;
      observation_coefficients.diag() /= 10;

      // Set up the regression coefficients and the predictors.
      regression_coefficients.randomize();
      predictors.randomize();

      build(residual_sd, df);
    }

    int nfactors() const {
      return observation_coefficients.ncol();
    }

    int nseries() const {
      return regression_coefficients.nrow();
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
      for (int factor = 0; factor < nfactors(); ++factor) {
        state(factor, 0) = rnorm();
        for (int time = 1; time < sample_size_ + test_size_; ++time) {
          state(factor, time) = state(factor, time - 1) + rnorm(0, factor_sd);
        }
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
      state_model.reset(new ConditionallyIndependentSharedLocalLevelStateModel(
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
          state_model_sampler)(
              state_model.get(),
              std::vector<Ptr<MvnBase>>(nseries(), slab),
              spikes,
              sigsq_params);
      state_model->set_method(state_model_sampler);
      state_model->set_initial_state_mean(state.col(0));
      state_model->set_initial_state_variance(SpdMatrix(nfactors(), 1.0));
      model->add_state(state_model);
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
    Matrix state;
    Matrix observation_coefficients;
    Matrix regression_coefficients;
    Matrix predictors;
    Matrix response;

    Ptr<StudentMvssRegressionModel> model;
    Ptr<ConditionallyIndependentSharedLocalLevelStateModel> state_model;
  };

};


#endif  //  BOOM_STATE_SPACE_STUDENT_MULTIVARIATE_TEST_FRAMEWORK_HPP_

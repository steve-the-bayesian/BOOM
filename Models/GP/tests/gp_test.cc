#include "gtest/gtest.h"

#include "Models/GP/GaussianProcessRegressionModel.hpp"
#include "Models/GP/PosteriorSamplers/MahalanobisKernelSampler.hpp"
#include "Models/GP/PosteriorSamplers/LinearMeanFunctionSampler.hpp"
#include "Models/GP/PosteriorSamplers/GaussianProcessRegressionPosteriorSampler.hpp"
#include "Models/GP/GpMeanFunction.hpp"
#include "Models/GP/kernels.hpp"
#include "Models/ChisqModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include "stats/moments.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class GpTest : public ::testing::Test {
   protected:
    GpTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(GpTest, MeanPredictionTest) {
    GaussianProcessRegressionModel model(
        new ZeroFunction,
        new RadialBasisFunction(.17),
        new UnivParams(49));

    int nobs = 20;

    Matrix X(nobs, 1);
    X.randomize();
    Vector y = 3 * X.col(0) + rnorm_vector(nobs, 4, 7);

    for (int i = 0; i < nobs; ++i) {
      NEW(RegressionData, data_point)(y[i], X.row(i));
      model.add_data(data_point);
    }

    int nnew = 5;
    Matrix Xnew(nnew, 1);
    Vector ynew = 3 * Xnew.col(0) + rnorm_vector(nnew, 4, 7);

    Ptr<MvnBase> predictive_distribution = model.predict_distribution(
        Xnew, true);
    Ptr<MvnBase> function_prediction = model.predict_distribution(
        Xnew, false);

    EXPECT_TRUE(VectorEquals(predictive_distribution->mu(),
                             function_prediction->mu()));

    for (int i = 0; i < predictive_distribution->dim(); ++i) {
      EXPECT_GT(predictive_distribution->Sigma()(i, i),
                function_prediction->Sigma()(i, i));
    }
  }

  // Verify that the log likelihood calculation is being done correctly.
  TEST_F(GpTest, LogLikelihood) {

    NEW(ZeroFunction, mean_param)();
    ZeroFunction &mean(*mean_param);
    NEW(RadialBasisFunction, kernel_param)(.57);
    RadialBasisFunction &kernel(*kernel_param);

    NEW(UnivParams, residual_variance_param)(square(10.2));

    GaussianProcessRegressionModel model(
        mean_param, kernel_param, residual_variance_param);

    int sample_size = 8;
    Matrix X(sample_size, 2);
    X.randomize();

    Vector mu(sample_size, 0.0);
    SpdMatrix Sigma(sample_size);
    for (int i = 0; i < sample_size; ++i) {
      mu[i] = mean(X.row(i));
      for (int j = 0; j < sample_size; ++j) {
        Sigma(i, j) = kernel(X.row(i), X.row(j));
      }
    }

    SpdMatrix residual_variance_matrix(sample_size);
    residual_variance_matrix.diag() = residual_variance_param->value();
    Vector y = rmvn(mu, Sigma + residual_variance_matrix);

    for (int i = 0; i < sample_size; ++i) {
      NEW(RegressionData, data_point)(y[i], X.row(i));
      model.add_data(data_point);
    }

    SpdMatrix Vinv = (Sigma + residual_variance_matrix).inv();

    EXPECT_NEAR(dmvn(y, mu, Vinv, true),
                model.log_likelihood(),
                1e-8);

  }

  // Check that MCMC for model parameters is
  TEST_F(GpTest, McmcTest_MahalanobisKernel) {
    int sample_size = 50;
    Matrix X(sample_size, 2);
    X.randomize();

    NEW(ZeroFunction, mean_param)();
    Vector mu = (*mean_param)(X);
    NEW(MahalanobisKernel, kernel_param)(X, 2.3);
    double true_kernel_scale = kernel_param->scale();

    SpdMatrix Sigma = (*kernel_param)(X);
    double true_residual_sd = 3.8;
    NEW(UnivParams, residual_variance_param)(square(true_residual_sd));
    SpdMatrix residual_variance_matrix(
        sample_size, residual_variance_param->value());

    Vector y = rmvn(mu, Sigma + residual_variance_matrix);

    NEW(GaussianProcessRegressionModel, model)(
        mean_param, kernel_param, residual_variance_param);

    for (int i = 0; i < sample_size; ++i) {
      NEW(RegressionData, data_point)(y[i], X.row(i));
      model->add_data(data_point);
    }

    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(ChisqModel, kernel_bandwidth_prior)(1.0, 1.0);

    NEW(GaussianProcessRegressionPosteriorSampler, sampler)(
        model.get(),
        new GP::NullSampler,
        new MahalanobisKernelSampler(
            kernel_param.get(), model.get(), kernel_bandwidth_prior),
        residual_precision_prior);
    model->set_method(sampler);

    int niter = 500;

    // Start the parameters from the wrong values.
    kernel_param->set_scale(.10);
    residual_variance_param->set(.05);

    Vector kernel_parameter_draws(niter);
    Vector residual_sd_draws(niter);
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      kernel_parameter_draws[i] = kernel_param->scale();
      residual_sd_draws[i] = model->residual_sd();
    }
    EXPECT_TRUE(CheckMcmcVector(kernel_parameter_draws, true_kernel_scale))
        << true_kernel_scale << " " << kernel_parameter_draws;
    EXPECT_GT(sd(kernel_parameter_draws), 0.0)
        << kernel_parameter_draws;

    EXPECT_TRUE(CheckMcmcVector(residual_sd_draws, true_residual_sd))
        << true_residual_sd << " " << residual_sd_draws;
    EXPECT_GT(sd(residual_sd_draws), 0.0) << residual_sd_draws;

    std::ofstream kernel_file("kernel_parameter_draws.out");
    kernel_file << true_kernel_scale << " " << kernel_parameter_draws;

    std::ofstream residual_sd_file("residual_sd_draws.out");
    residual_sd_file << true_residual_sd << " " << residual_sd_draws;
  }

  // Test that the linear mean function works as expected, and that the
  // posterior sampler for the linear mean function recovers the parameters.
  TEST_F(GpTest, TestLinearMeanFunction) {

    // ----------------------------------------------------------------------
    // Simulate the data
    // ----------------------------------------------------------------------
    int sample_size = 100;
    int xdim = 3;
    Matrix X(sample_size, xdim);
    X.randomize();

    Vector beta = {1.0, -2.0, 3.0};
    Vector y = X * beta + rnorm_vector(sample_size, 0.0, 0.3);

    // ----------------------------------------------------------------------
    // Build the model objects.
    // ----------------------------------------------------------------------
    NEW(GlmCoefs, mean_function_coefs)(beta, 1.0);
    NEW(LinearMeanFunction, mean_function)(mean_function_coefs);

    NEW(RadialBasisFunction, kernel)(1.0);
    NEW(UnivParams, residual_variance)(1.0);
    NEW(GaussianProcessRegressionModel, model)(
        mean_function, kernel, residual_variance);

    // ----------------------------------------------------------------------
    // Add the data to the model.
    // ----------------------------------------------------------------------
    for (int i = 0; i < sample_size; ++i) {
      NEW(RegressionData, data_point)(y[i], X.row(i));
      model->add_data(data_point);
    }

    // ----------------------------------------------------------------------
    // Set the priors.
    // ----------------------------------------------------------------------
    NEW(MvnModel, beta_prior)(Vector(xdim, 0.0), SpdMatrix(xdim, 1.0));
    NEW(ChisqModel, residual_precision_prior)(0.3, 1.0);
    NEW(GaussianProcessRegressionPosteriorSampler, sampler)(
        model.get(),
        new LinearMeanFunctionSampler(
            mean_function.get(), model.get(), beta_prior),
        new GP::NullSampler,
        residual_precision_prior);
    model->set_method(sampler);

    // ----------------------------------------------------------------------
    // Run the MCMC
    // ----------------------------------------------------------------------
    int niter = 500;
    Matrix beta_draws(niter, xdim);
    Matrix fun_draws(niter, sample_size);
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      beta_draws.row(i) = mean_function_coefs->Beta();

    }

    auto status = CheckMcmcMatrix(beta_draws, beta);
    EXPECT_TRUE(status.ok) << status.error_message();

    std::ofstream out("beta_draws.out");
    out << beta << "\n" << beta_draws;
  }

}  // namespace

#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"

#include "stats/moments.hpp"
#include "cpputil/lse.hpp"
#include "LinAlg/Cholesky.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class RegressionModelTest : public ::testing::Test {
   protected:
    RegressionModelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(RegressionModelTest, RegSufTest) {
    int n = 100;
    int p = 3;
    Matrix X(n, p);
    X.randomize();
    X.col(0) = 1.0;

    Vector y(n);
    y.randomize();

    double tiny = 1e-8;

    NeRegSuf suf(X, y);
    EXPECT_TRUE(MatrixEquals(suf.xtx(), t(X) * X));
    EXPECT_TRUE(VectorEquals(suf.xty(), t(X) * y));
    EXPECT_EQ(suf.n(), n);

    EXPECT_NEAR(suf.ybar(), mean(y), tiny);
    EXPECT_NEAR(suf.ybar(), mean(y), tiny);

    EXPECT_NEAR(suf.sample_sd(), sd(y), tiny);
  }

  TEST_F(RegressionModelTest, DataConstructor) {
    int nobs = 1000;
    int nvars = 10;
    double residual_sd = 1.3;
    Matrix predictors(nobs, nvars);
    predictors.randomize();
    predictors.col(0) = 1.0;
    Vector coefficients(nvars);
    coefficients.randomize();
    for (int i = 6; i < nvars; ++i) {
      coefficients[i] = 0.0;
    }
    Vector response = predictors * coefficients;
    for (int i = 0; i < nobs; ++i) {
      response[i] += rnorm(0, residual_sd);
    }
    NEW(RegressionModel, model)(predictors, response);
  }

  TEST_F(RegressionModelTest, McmcTest) {
    int nobs = 1000;
    int nvars = 10;
    double residual_sd = 1.3;
    Matrix predictors(nobs, nvars);
    predictors.randomize();
    predictors.col(0) = 1.0;
    Vector coefficients(nvars);
    coefficients.randomize();
    for (int i = 6; i < nvars; ++i) {
      coefficients[i] = 0.0;
    }
    Vector response = predictors * coefficients;
    for (int i = 0; i < nobs; ++i) {
      response[i] += rnorm(0, residual_sd);
    }
    NEW(RegressionModel, model)(predictors, response);

    NEW(MvnGivenScalarSigma, coefficient_prior)(
        model->suf()->xtx() / model->suf()->n(), model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, residual_sd);
    NEW(RegressionConjSampler, sampler)(
        model.get(), coefficient_prior, residual_precision_prior);
    model->set_method(sampler);

    int niter = 500;
    Matrix beta_draws(niter, model->xdim());
    Vector sigma_draws(niter);
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
    }

    EXPECT_TRUE(CheckMcmcVector(sigma_draws, 1.3));
    EXPECT_EQ("", CheckWithinRage(sigma_draws, 0, 10));
    auto status = CheckMcmcMatrix(beta_draws, coefficients);
    EXPECT_TRUE(status.ok) << status;
  }

  // Verify that RegressionModel::log_likelihood matches the direct log
  // likelihood calculation, and that RegressionModel::marginal_log_likelihood
  // corresponds to the log of the expected likelihood function, integrating
  // beta over a prior.
  TEST_F(RegressionModelTest, LoglikeTest) {
    // Test log likelihood and marginal loglike.
    int sample_size = 1000;
    int xdim = 3;
    Matrix X(sample_size, xdim);
    X.randomize();
    X.col(0) = 1.0;

    Vector b0 = {10, -2, 5};
    Matrix S0("1.0 0.6 0.4 | 0.6 1.5 0.8 | 0.4 0.8 2.0");
    SpdMatrix Sigma(S0);

    double residual_sd = 1.20;
    double residual_variance = residual_sd * residual_sd;
    Vector beta = rmvn(b0, Sigma);
    Vector yhat = X * beta;
    Vector y = yhat + rnorm_vector(sample_size, 0, residual_sd);

    SpdMatrix xtx = X.inner();
    Vector xty = X.Tmult(y);
    double yty = y.dot(y);

    NeRegSuf suf(xtx, xty, yty, sample_size, mean(y), mean(X));

    // Test loglike.
    double direct_loglike = 0;
    for (int i = 0; i < y.size(); ++i) {
      direct_loglike += dnorm(y[i], yhat[i], residual_sd, true);
    }
    EXPECT_NEAR(direct_loglike,
                RegressionModel::log_likelihood(beta, residual_variance, suf),
                1e-4);

    // Test marginal loglike
    int niter = 10000;
    Vector draws(niter);
    for (int i = 0; i < niter; ++i) {
      beta = rmvn(b0, residual_variance * Sigma);
      draws[i] = RegressionModel::log_likelihood(beta, residual_variance, suf);
    }
    double monte_carlo_marginal_loglike = lse(draws) - log(niter);

    SpdMatrix siginv = Sigma.inv();
    SpdMatrix unscaled_posterior_precision = xtx + siginv;
    Cholesky unscaled_posterior_precision_chol(unscaled_posterior_precision);
    Vector posterior_mean = unscaled_posterior_precision_chol.solve(
        siginv * b0 + xty);

    double marginal_loglike = RegressionModel::marginal_log_likelihood(
        residual_variance,
        xtx,
        xty,
        yty,
        sample_size,
        b0,
        siginv.chol(),
        posterior_mean,
        unscaled_posterior_precision_chol.getL());

    EXPECT_NEAR(marginal_loglike, monte_carlo_marginal_loglike, 3.0);
  }


}  // namespace

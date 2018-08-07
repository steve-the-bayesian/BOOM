#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"
#include "LinAlg/Cholesky.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultivariateRegressionTest : public ::testing::Test {
   protected:
    MultivariateRegressionTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MultivariateRegressionTest, LogLikelihood) {
    int xdim = 2;
    int ydim = 3;
    Matrix coefficients(xdim, ydim);
    coefficients.randomize();
    SpdMatrix Sigma(ydim);
    Sigma.randomize();
    SpdMatrix Siginv = Sigma.inv();
    double ldsi = Siginv.logdet();
    SpdMatrix SSE(ydim, 0.0);
    
    int nobs = 10;
    Matrix predictors(nobs, xdim);
    predictors.randomize();
    Matrix response(nobs, ydim);

    double loglike_direct = 0;
    double qform = 0;
    MultivariateRegressionModel model(2, 3);
    for (int i = 0; i < nobs; ++i) {
      Vector yhat = predictors.row(i) * coefficients;
      response.row(i) = rmvn(yhat, Sigma);
      NEW(MvRegData, data_point)(response.row(i), predictors.row(i));
      model.add_data(data_point);
      loglike_direct += dmvn(response.row(i), yhat, Siginv, ldsi, true);
      SSE.add_outer(response.row(i) - yhat);
      qform += Siginv.Mdist(response.row(i), yhat);
    }

    EXPECT_TRUE(MatrixEquals(SSE, model.suf()->SSE(coefficients)));
    EXPECT_NEAR(qform, trace(SSE * Siginv), 1e-5);
    
    double loglike = model.log_likelihood(coefficients, Sigma);
    double loglike_inv = model.log_likelihood_ivar(coefficients, Siginv);
    EXPECT_NEAR(loglike, loglike_inv, 1e-6);
    EXPECT_NEAR(loglike, loglike_direct, 1e-6);
  }

  TEST_F(MultivariateRegressionTest, McmcConjugatePrior) {
    int xdim = 2;
    int ydim = 3;
    Matrix coefficients(xdim, ydim);
    coefficients.randomize();
    int nobs = 1000;
    SpdMatrix Sigma(ydim);
    Sigma.randomize();

    MultivariateRegressionModel model(xdim, ydim);
    
    for (int i = 0; i < nobs; ++i) {
      Vector predictors(xdim);
      predictors.randomize();
      Vector yhat = predictors * coefficients;
      Vector response = rmvn(yhat, Sigma);
      NEW(MvRegData, data_point)(response, predictors);
      model.add_data(data_point);
    }

    NEW(MultivariateRegressionSampler, sampler)(
        &model,
        coefficients,
        1.0,
        1.0,
        Sigma);
    model.set_method(sampler);

    int niter = 1e+4;
    Matrix beta_draws(niter, xdim * ydim);
    Matrix sigma_draws(niter, ydim * (ydim + 1) / 2);
    Vector true_beta = vec(coefficients);
    Vector true_sigma = Sigma.vectorize();
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      beta_draws.row(i) = vec(model.Beta());
      sigma_draws.row(i) = model.Sigma().vectorize();
    }

    auto status = CheckMcmcMatrix(beta_draws, true_beta);
    EXPECT_TRUE(status.ok) << status.error_message();

    status = CheckMcmcMatrix(sigma_draws, true_sigma);
    EXPECT_TRUE(status.ok) << status.error_message();
  }
  
}  // namespace

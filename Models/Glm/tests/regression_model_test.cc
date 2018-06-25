#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"

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
  
}  // namespace

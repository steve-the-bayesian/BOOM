#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/AdaptiveSpikeSlabRegressionSampler.hpp"

#include "test_utils/test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "stats/ECDF.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class RegressionSpikeSlabTest : public ::testing::Test {
   protected:
    RegressionSpikeSlabTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(RegressionSpikeSlabTest, Small) {
    int nobs = 1000;
    int nvars = 10;
    int niter = 1000;
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
    SpdMatrix xtx = model->suf()->xtx();
    NEW(MvnGivenScalarSigma, slab)(
        Vector(nvars, 0), xtx / nobs, model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(nvars, .5);
    NEW(AdaptiveSpikeSlabRegressionSampler, sampler)(
        model.get(), slab, residual_precision_prior, spike);
    model->set_method(sampler);
    Vector sigma_draws(niter);
    Matrix beta_draws(niter, nvars);
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
    }

    NEW(RegressionModel, model2)(predictors, response);
    NEW(MvnGivenScalarSigma, slab2)(
        Vector(nvars, 0), xtx / nobs, model2->Sigsq_prm());
    NEW(BregVsSampler, old_sampler)(
        model2.get(), slab2, residual_precision_prior, spike);
    model2->set_method(old_sampler);
    
    Vector more_sigma_draws(niter);
    Matrix more_beta_draws(niter, nvars);
    for (int i = 0; i < niter; ++i) {
      model2->sample_posterior();
      more_sigma_draws[i] = model2->sigma();
      more_beta_draws.row(i) = model2->Beta();
    }

    EXPECT_TRUE(EquivalentSimulations(sigma_draws, more_sigma_draws))
        << cbind(sigma_draws, more_sigma_draws);

    EXPECT_TRUE(EquivalentSimulations(beta_draws.col(3),
                                      more_beta_draws.col(3)))
        << cbind(beta_draws, more_beta_draws)
        << AsciiDistributionCompare(beta_draws.col(3),
                                    more_beta_draws.col(3));


    for (int i = 0; i < nvars; ++i) {
      EXPECT_TRUE(EquivalentSimulations(beta_draws.col(i),
                                        more_beta_draws.col(i)));
    }
  }

  TEST_F(RegressionSpikeSlabTest, Large) {
    int nobs = 10000;
    int nvars = 1000;
    int niter = 1000;
    double residual_sd = 0.3;
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

    NEW(RegressionModel, model)(predictors.ncol());
    model->coef().drop_all();
    model->coef().add(0);
    
    for (int i = 0; i < predictors.nrow(); ++i) {
      NEW(RegressionData, dp)(response[i], predictors.row(i));
      model->add_data(dp);
    }
    SpdMatrix xtx = model->suf()->xtx();
    NEW(MvnGivenScalarSigma, slab)(
        Vector(nvars, 0), xtx / nobs, model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(nvars, 5.0 / nvars);
    NEW(AdaptiveSpikeSlabRegressionSampler, sampler)(
        model.get(), slab, residual_precision_prior, spike);
    model->set_method(sampler);
    Vector sigma_draws(niter);
    Matrix beta_draws(niter, nvars);
    Vector size_draws(niter);
    for (int i = 0; i < niter; ++i) {
      std::cout << "iteration " << i << std::endl;
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
      size_draws[i] = model->coef().inc().nvars();
    }

    ECDF size_distribution(size_draws);
    EXPECT_LT(size_distribution(10), .05);
  }
  
}  // namespace

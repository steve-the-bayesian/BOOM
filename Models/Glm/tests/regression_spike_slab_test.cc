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
    RegressionSpikeSlabTest()
        : nobs_(1000),
          xdim_(10),
          niter_(1000),
          residual_sd_(0.3)
    {
      GlobalRng::rng.seed(8675309);
    }

    void SimulateCoefficients() {
      coefficients_.resize(xdim_);
      coefficients_.randomize();
      VectorView zero_coefficients(coefficients_, 6);
      zero_coefficients = 0.0;
    }
    
    void SimulatePredictors() {
      predictors_.resize(nobs_, xdim_);
      predictors_.randomize();
      predictors_.col(0) = 1.0;
    }

    void SimulateResponse() {
      response_ = predictors_ * coefficients_;
      for (int i = 0; i < response_.size(); ++i) {
        response_[i] += rnorm_mt(GlobalRng::rng, 0, residual_sd_);
      }
    }

    int nobs_;
    int xdim_;
    int niter_;
    Matrix predictors_;
    Vector response_;
    double residual_sd_;
    Vector coefficients_;
  };

  inline double inclusion_probability(const ConstVectorView &coefficients) {
    double ans = 0;
    for (auto y : coefficients) {
      ans += (y != 0);
    }
    return ans / coefficients.size();
  }
  
  TEST_F(RegressionSpikeSlabTest, Small) {
    SimulatePredictors();
    SimulateCoefficients();
    VectorView(coefficients_, 6) = 0.0;
    SimulateResponse();

    NEW(RegressionModel, model)(predictors_, response_);
    SpdMatrix xtx = model->suf()->xtx();
    NEW(MvnGivenScalarSigma, slab)(
        Vector(xdim_, 0), xtx / nobs_, model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(xdim_, .5);
    NEW(AdaptiveSpikeSlabRegressionSampler, sampler)(
        model.get(), slab, residual_precision_prior, spike);
    model->set_method(sampler);
    Vector sigma_draws(niter_);
    Matrix beta_draws(niter_, xdim_);
    for (int i = 0; i < niter_; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
    }

    NEW(RegressionModel, model2)(predictors_, response_);
    NEW(MvnGivenScalarSigma, slab2)(
        Vector(xdim_, 0), xtx / nobs_, model2->Sigsq_prm());
    NEW(BregVsSampler, old_sampler)(
        model2.get(), slab2, residual_precision_prior, spike);
    model2->set_method(old_sampler);
    
    Vector more_sigma_draws(niter_);
    Matrix more_beta_draws(niter_, xdim_);
    for (int i = 0; i < niter_; ++i) {
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

    for (int i = 0; i < xdim_; ++i) {
      EXPECT_TRUE(EquivalentSimulations(beta_draws.col(i),
                                        more_beta_draws.col(i)));
    }
  }

  // Check that setting max_size actually constrains the model to that many
  // coefficients.
  TEST_F(RegressionSpikeSlabTest, TestMaxSizeControl) {
    xdim_ = 10;
    SimulatePredictors();
    coefficients_.resize(xdim_);
    coefficients_ = 0.0;
    coefficients_[0] = 23;
    coefficients_[1] = 17;
    coefficients_[2] = 12;
    coefficients_[3] = -33;
    residual_sd_ = 0.1;
    SimulateResponse();

    // The model has 4 obviously very large coefficients.  Setting max_size = 2
    // should constrain posterior distribution to only allowing 2.
    int max_size = 2; 

    NEW(RegressionModel, model)(predictors_, response_, false);
    model->coef().drop_all();
    model->coef().add(0);

    SpdMatrix xtx = model->suf()->xtx();
    NEW(MvnGivenScalarSigma, slab)(
        Vector(xdim_, 0), xtx / nobs_, model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(xdim_, 5.0 / xdim_);

    spike->set_max_model_size(max_size);  // <-------------- 
    
    NEW(BregVsSampler, sampler)(
        model.get(), slab, residual_precision_prior, spike);
    
    model->set_method(sampler);
    Vector sigma_draws(niter_);
    Matrix beta_draws(niter_, xdim_);
    Vector size_draws(niter_);
    for (int i = 0; i < niter_; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
      size_draws[i] = model->coef().inc().nvars();
    }

    ECDF size_distribution(size_draws);
    EXPECT_LE(size_distribution.quantile(1.0), 2);

    // std::ofstream beta_out("beta_draws.out");
    // beta_out << beta_draws;
  }
  
  TEST_F(RegressionSpikeSlabTest, Large) {
    xdim_ = 100;
    SimulatePredictors();
    SimulateCoefficients();
    SimulateResponse();

    NEW(RegressionModel, model)(predictors_, response_, false);
    model->coef().drop_all();
    model->coef().add(0);
    
    SpdMatrix xtx = model->suf()->xtx();
    NEW(MvnGivenScalarSigma, slab)(
        Vector(xdim_, 0), xtx / nobs_, model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(xdim_, 5.0 / xdim_);
    NEW(AdaptiveSpikeSlabRegressionSampler, sampler)(
        model.get(), slab, residual_precision_prior, spike);
    model->set_method(sampler);
    Vector sigma_draws(niter_);
    Matrix beta_draws(niter_, xdim_);
    Vector size_draws(niter_);
    for (int i = 0; i < niter_; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
      size_draws[i] = model->coef().inc().nvars();
    }

    ECDF size_distribution(size_draws);
    // The fraction of models with 8 or fewer included predictors should be
    // high.
    EXPECT_GE(size_distribution(8), .95);
  }

  TEST_F(RegressionSpikeSlabTest, PerfectCollinearity) {
    xdim_ = 50;
    niter_ *= 5;
    SimulatePredictors();
    predictors_.col(2) = 1.7 * predictors_.col(1);
    predictors_.col(3) = -2.4 * predictors_.col(2);

    // Pick some obvious values for coefficients that the model should be able
    // to find easily.  Because of the perfect correlation, coefficient 1, 2,
    // and 3 each have about a 1/3 chance of appearing.
    coefficients_ = Vector(xdim_, 0.0);
    coefficients_[0] = 12;
    coefficients_[1] = 18;
    coefficients_[2] = 0;
    coefficients_[3] = 0;
    coefficients_[4] = -4;
    coefficients_[5] = 73;
    
    SimulateResponse();

    NEW(RegressionModel, model)(predictors_, response_, false);
    model->coef().drop_all();
    model->coef().add(0);
    SpdMatrix xtx = model->suf()->xtx();
    NEW(MvnGivenScalarSigma, slab)(
        Vector(xdim_, 0), xtx / nobs_, model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(xdim_, 5.0 / xdim_);
    NEW(BregVsSampler, sampler)(
        model.get(), slab, residual_precision_prior, spike);
    model->set_method(sampler);

    Matrix beta_draws(niter_, xdim_);
    Vector sigma_draws(niter_);
    
    for (int i = 0; i < niter_; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
    }

    EXPECT_GE(inclusion_probability(beta_draws.col(0)), .9);
    EXPECT_GE(inclusion_probability(beta_draws.col(1)), .20);
    EXPECT_LE(inclusion_probability(beta_draws.col(1)), .40);
    EXPECT_GE(inclusion_probability(beta_draws.col(2)), .20);
    EXPECT_LE(inclusion_probability(beta_draws.col(2)), .40);
    EXPECT_GE(inclusion_probability(beta_draws.col(3)), .20);
    EXPECT_LE(inclusion_probability(beta_draws.col(3)), .40);
    EXPECT_GE(inclusion_probability(beta_draws.col(4)), .9);
    EXPECT_GE(inclusion_probability(beta_draws.col(5)), .9);
  }
  
}  // namespace

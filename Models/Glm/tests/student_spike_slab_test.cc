#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/TRegression.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/PosteriorSamplers/TRegressionSpikeSlabSampler.hpp"

#include "test_utils/test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "stats/ECDF.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class StudentSpikeSlabTest : public ::testing::Test {
   protected:
    StudentSpikeSlabTest()
        : nobs_(1000),
          xdim_(10),
          niter_(1000),
          residual_sd_(0.3),
          tail_thickness_(5)
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
        response_[i] += rstudent_mt(GlobalRng::rng, 0, residual_sd_, tail_thickness_);
      }
    }

    int nobs_;
    int xdim_;
    int niter_;
    Matrix predictors_;
    Vector response_;
    double residual_sd_;
    double tail_thickness_;
    Vector coefficients_;
  };

  inline double inclusion_probability(const ConstVectorView &coefficients) {
    double ans = 0;
    for (auto y : coefficients) {
      ans += (y != 0);
    }
    return ans / coefficients.size();
  }

  // Simulate fake data with only the first 5 coefficients nonzero.
  TEST_F(StudentSpikeSlabTest, Small) {
    SimulatePredictors();
    SimulateCoefficients();
    VectorView(coefficients_, 6) = 0.0;
    SimulateResponse();

    NEW(TRegressionModel, model)(predictors_, response_);
    NEW(RegressionModel, reg)(predictors_, response_);
    SpdMatrix xtx = reg->suf()->xtx();

    NEW(MvnGivenScalarSigma, slab)(
        Vector(xdim_, 0), xtx / nobs_, model->Sigsq_prm());
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(xdim_, .5);
    NEW(ChisqModel, tail_thickness_prior)(5.0, 1.0);
    NEW(TRegressionSpikeSlabSampler, sampler)(
        model.get(), slab, spike,
        residual_precision_prior,
        tail_thickness_prior);
    model->set_method(sampler);
    Vector sigma_draws(niter_);
    Vector nu_draws(niter_);
    Matrix beta_draws(niter_, xdim_);
    for (int i = 0; i < niter_; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
      nu_draws[i] = model->nu();
    }

    EXPECT_TRUE(CheckMcmcVector(nu_draws, tail_thickness_))
        << "Tail thickness parameter failed to cover.";

    EXPECT_TRUE(CheckMcmcVector(sigma_draws, residual_sd_))
        << "Residual SD parameter failed to cover.";
  }

}  // namespace

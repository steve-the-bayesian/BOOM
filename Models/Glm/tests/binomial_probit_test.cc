#include "gtest/gtest.h"

#include "Models/MvnModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/BinomialProbitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialProbitDataImputer.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialProbitCompositeSpikeSlabSampler.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class BinomialProbitTest : public ::testing::Test {
   protected:
    BinomialProbitTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(BinomialProbitTest, DataImputer) {
    BinomialProbitDataImputer imputer;
    for (int i = 0; i < 100; ++i) {
      imputer.impute(GlobalRng::rng, 3, 0, 1.2);
    }

    // Test zero trials.
    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 0, 0, 1.2);
      EXPECT_DOUBLE_EQ(ans, 0.0);
    }

    // Test certain negative.
    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 3, 0, -100.8);
      EXPECT_LT(ans, 0.0);
    }

    // Test forced negative.
    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 3, 0, 100.8);
      EXPECT_LT(ans, 0.0);
    }

    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 3, 3, 100.7);
      EXPECT_GT(ans, 0.0);
    }
    
    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 3, 3, -100.2);
      EXPECT_GT(ans, 0.0);
    }

    // Test large sample sizes

    // Test certain negative.
    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 30, 0, -100.8);
      EXPECT_LT(ans, 0.0);
    }

    // Test forced negative.
    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 30, 0, 100.8);
      EXPECT_LT(ans, 0.0);
    }

    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 30, 30, 100.7);
      EXPECT_GT(ans, 0.0);
    }
    
    for (int i = 0; i < 100; ++i) {
      double ans = imputer.impute(GlobalRng::rng, 30, 30, -100.2);
      EXPECT_GT(ans, 0.0);
    }
  }

  TEST_F(BinomialProbitTest, SpikeSlabSamplerTest) {
    Matrix predictors(100, 5);
    predictors.randomize();
    predictors.col(0) = 1.0;
    Vector beta(5);
    beta[0] = -2;
    beta.randomize();

    Vector eta = predictors * beta;
    NEW(BinomialProbitModel, model)(predictors.ncol());
    for (int i = 0; i < eta.size(); ++i) {
      double prob = pnorm(eta[i]);
      int n = rpois(5.0);
      int y = rbinom(n, prob);
      NEW(BinomialRegressionData, data_point)(y, n, predictors.row(i));
      model->add_data(data_point);
    }

    int clt_threshold = 10;
    double proposal_df = 3.0;
    int xdim = predictors.ncol();
    Vector prior_mean(xdim, 0.0);
    SpdMatrix prior_precision = predictors.inner() / predictors.nrow();
    NEW(MvnModel, slab)(prior_mean, prior_precision.inv());
    NEW(VariableSelectionPrior, spike)(xdim, 1.0 / xdim);
    NEW(BinomialProbitCompositeSpikeSlabSampler, sampler)(
        model.get(),
        slab,
        spike,
        clt_threshold,
        proposal_df);
    model->set_method(sampler);

    for (int i = 0; i < 100; ++i) {
      model->sample_posterior();
    }
  }
  
}  // namespace

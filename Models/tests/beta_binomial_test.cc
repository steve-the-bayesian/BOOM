#include "gtest/gtest.h"
#include "Models/BetaBinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/UniformModel.hpp"

#include "Models/PosteriorSamplers/BetaBinomialPosteriorSampler.hpp"

#include "stats/FreqDist.hpp"
#include "stats/ChiSquareTest.hpp"

#include "distributions.hpp"
#include "Bmath/Bmath.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using Rmath::lgammafn;

  using namespace BOOM;
  using std::endl;
  using std::cout;

  class BetaBinomialTest : public ::testing::Test {
   protected:
    BetaBinomialTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  double log_nc(double n, double y) {
    return lgammafn(n + 1) - lgammafn(y + 1) - lgammafn(n - y + 1);
  }

  TEST_F(BetaBinomialTest, Suf) {
    BetaBinomialSuf suf;
    EXPECT_DOUBLE_EQ(0.0, suf.log_normalizing_constant());

    suf.add_data(1, 0, 1);
    EXPECT_EQ(1, suf.count_table().size());
    EXPECT_EQ(1, suf.sample_size());
    EXPECT_NEAR(suf.log_normalizing_constant(),
                log_nc(1, 0), 1e-8);

    suf.add_data(1, 0, 5);
    EXPECT_EQ(1, suf.count_table().size());
    EXPECT_EQ(6, suf.sample_size());
    EXPECT_NEAR(suf.log_normalizing_constant(),
                6 * log_nc(1, 0),
                1e-8);

    suf.add_data(1, 1, 3);
    EXPECT_EQ(2, suf.count_table().size());
    EXPECT_EQ(9, suf.sample_size());
    EXPECT_NEAR(suf.log_normalizing_constant(),
                6 * log_nc(1, 0) + 3 * log_nc(1, 1),
                1e-8);

    suf.add_data(7, 4, 3);
    EXPECT_EQ(3, suf.count_table().size());
    EXPECT_EQ(12, suf.sample_size());
    EXPECT_NEAR(suf.log_normalizing_constant(),
                6 * log_nc(1, 0) + 3 * log_nc(1, 1) + 3 * log_nc(7, 4),
                1e-8);
  }

  // Check that the beta binomial probability mass function is coded correctly.
  // Simulate a bunch of draws from the distribution and compare the empirical
  // draws to the true distribution.
  TEST_F(BetaBinomialTest, PmfTest) {
    double a = 3.0;
    double b = 12.0;
    BetaBinomialModel model(3, 12);
    int N = 30;

    int sample_size = 10000;
    std::vector<int> draws(sample_size);
    for (int i = 0; i < sample_size; ++i) {
      draws[i] = model.sim(GlobalRng::rng, N);
    }
    FrequencyDistribution empirical(draws, 0, N);

    Vector true_distribution(N + 1);
    for (int i = 0; i <= N; ++i) {
      true_distribution[i] = exp(BetaBinomialModel::logp(N, i, a, b));
    }
    EXPECT_NEAR(true_distribution.sum(), 1.0, 1e-6);

    OneWayChiSquareTest chisq_test(empirical, true_distribution);
    EXPECT_GT(chisq_test.p_value(), 0.05);
  }

  // Check that the log likelihood function matches
  TEST_F(BetaBinomialTest, LogLikelihoodTest) {
    BetaBinomialModel model(1.0, 2.0);

    std::vector<Ptr<BinomialData>> data;
    double a = 1.3;
    double b = 2.7;

    for (int i = 0; i < 1; ++i) {
      int N = 12;
      int y = 7;
      NEW(BinomialData, dp)(N, y);
      model.add_data(dp);
      data.push_back(dp);
    }
    double model_loglike = model.loglike(a, b);
    double raw_loglike = BetaBinomialModel::logp(12, 7, a, b);
    EXPECT_NEAR(model_loglike, raw_loglike, 1e-8);
    EXPECT_NEAR(model.suf()->log_normalizing_constant(),
                lgammafn(12 + 1) - lgammafn(7 + 1) - lgammafn(5 + 1),
                1e-8);
    EXPECT_EQ(model.suf()->sample_size(), 1);

    for (int i = 0; i < 1000; ++i) {
      int N = rpois(10);
      int y = rbinom(N, .4);
      NEW(BinomialData, dp)(N, y);
      data.push_back(dp);
      model.add_data(dp);
    }

    model_loglike = model.loglike(a, b);
    raw_loglike = 0;
    for (int i = 0; i < data.size(); ++i) {
      Ptr<BinomialData> dp = data[i];
      raw_loglike += BetaBinomialModel::logp(dp->n(), dp->y(), a, b);
    }
    EXPECT_NEAR(model_loglike, raw_loglike, 1e-8);
  }

  TEST_F(BetaBinomialTest, McmcTest) {
    double a = 3.8;
    double b = 15.7;
    int sample_size = 1000;
    int niter=2500;

    NEW(BetaBinomialModel, model)(1.0, 1.0);

    for (int i = 0; i < sample_size; ++i) {
      int N = rpois(10);
      double p  = rbeta(a, b);
      int y = rbinom(N, p);
      model->suf()->add_data(N, y, 1);
    }

    NEW(BetaModel, mean_prior)(1, 1);
    NEW(UniformModel, sample_size_prior)(.1, 100);
    NEW(BetaBinomialPosteriorSampler, sampler)(
        model.get(), mean_prior, sample_size_prior);
    sampler->set_sampling_method(BetaBinomialPosteriorSampler::SLICE);
    model->set_method(sampler);

    Vector a_draws(niter);
    Vector b_draws(niter);

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      a_draws[i] = model->a();
      b_draws[i] = model->b();
    }

    EXPECT_TRUE(CheckMcmcVector(a_draws, a, .95, "a.draws"));
    EXPECT_TRUE(CheckMcmcVector(b_draws, b, .95, "b.draws"));
  }

}  // namespace

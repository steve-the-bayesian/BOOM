#include "gtest/gtest.h"

#include "Bandits/BinomialBandit.hpp"

#include "Models/BinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/PosteriorSamplers/BetaBinomialSampler.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class BinomialBanditTest : public ::testing::Test {
   protected:
    BinomialBanditTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(BinomialBanditTest, SmokeTest) {
  }

  TEST_F(BinomialBanditTest, ValueTest) {
    std::vector<Ptr<BinomialModel>> models;
    models.push_back(new BinomialModel(.3));
    models.push_back(new BinomialModel(.5));
    models.push_back(new BinomialModel(.1));

    BinomialBandit bandit(models);
    
    EXPECT_DOUBLE_EQ(bandit.Value(0), .3);
    EXPECT_DOUBLE_EQ(bandit.Value(1), .5);
    EXPECT_DOUBLE_EQ(bandit.Value(2), .1);

    NEW(VectorParams, probs)({.1, .2, .7});
    
    EXPECT_DOUBLE_EQ(bandit.Value(0, probs.get()), .1);
    EXPECT_DOUBLE_EQ(bandit.Value(1, probs.get()), .2);
    EXPECT_DOUBLE_EQ(bandit.Value(2, probs.get()), .7);
  }

  TEST_F(BinomialBanditTest, ObserveDataTest) {
    std::vector<Ptr<BinomialModel>> models;
    models.push_back(new BinomialModel(.3));
    models.push_back(new BinomialModel(.5));
    models.push_back(new BinomialModel(.1));

    BinomialBandit bandit(models);

    bandit.ObserveData(0, 3, 7);
    bandit.ObserveData(1, 4, 6);
    bandit.ObserveData(2, 2, 9);

    bandit.ObserveData(0, 0, 0);
  }

  TEST_F(BinomialBanditTest, UpdatePosteriorTest) {
    std::vector<Ptr<BinomialModel>> models;
    models.push_back(new BinomialModel(.3));
    models.push_back(new BinomialModel(.5));
    models.push_back(new BinomialModel(.1));

    for (auto &model : models) {
      NEW(BetaModel, prior)(1, 1);
      NEW(BetaBinomialSampler, sampler)(model.get(), prior);
      model->set_method(sampler);
    }

    BinomialBandit bandit(models);
    bandit.ObserveData(0, 300, 700);
    bandit.ObserveData(1, 400, 600);
    bandit.ObserveData(2, 200, 900);

    bandit.UpdatePosterior(1000);

    EXPECT_GT(bandit.Value(1), bandit.Value(0));
    EXPECT_GT(bandit.Value(1), bandit.Value(2));
    EXPECT_GT(bandit.Value(0), bandit.Value(2));

    for (auto &model : models) {
      model->clear_data();
    }
    
    bandit.ObserveData(0, 3, 7);
    bandit.ObserveData(1, 4, 6);
    bandit.ObserveData(2, 2, 9);
    int ndraws = 100000;
    bandit.UpdatePosterior(ndraws);

    EXPECT_EQ(3, bandit.OptimalArmProbabilities().size());
    
    EXPECT_GT(bandit.OptimalArmProbabilities()[0], .15);
    EXPECT_LT(bandit.OptimalArmProbabilities()[0], .25);

    EXPECT_GT(bandit.OptimalArmProbabilities()[1], .7);
    EXPECT_LT(bandit.OptimalArmProbabilities()[1], .8);
    
    EXPECT_GT(bandit.OptimalArmProbabilities()[2], .01);
    EXPECT_LT(bandit.OptimalArmProbabilities()[2], .04);

    const Vector &value(bandit.ValueRemainingDistribution());
    EXPECT_EQ(value.size(), ndraws);
  }

}  // namespace

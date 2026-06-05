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
    
    EXPECT_DOUBLE_EQ(bandit.value(0), .3);
    EXPECT_DOUBLE_EQ(bandit.value(1), .5);
    EXPECT_DOUBLE_EQ(bandit.value(2), .1);

  }

  TEST_F(BinomialBanditTest, ObserveDataTest) {
    std::vector<Ptr<BinomialModel>> models;
    models.push_back(new BinomialModel(.3));
    models.push_back(new BinomialModel(.5));
    models.push_back(new BinomialModel(.1));

    BinomialBandit bandit(models);

    bandit.observe_data(0, 3, 7);
    bandit.observe_data(1, 4, 6);
    bandit.observe_data(2, 2, 9);

    bandit.observe_data(0, 0, 0);
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
    bandit.observe_data(0, 300, 700);
    bandit.observe_data(1, 400, 600);
    bandit.observe_data(2, 200, 900);

    bandit.update_posterior(1000);

    EXPECT_GT(bandit.value(1), bandit.value(0));
    EXPECT_GT(bandit.value(1), bandit.value(2));
    EXPECT_GT(bandit.value(0), bandit.value(2));

    for (auto &model : models) {
      model->clear_data();
    }
    
    bandit.observe_data(0, 3, 7);
    bandit.observe_data(1, 4, 6);
    bandit.observe_data(2, 2, 9);
    int ndraws = 100000;
    bandit.update_posterior(ndraws);

    EXPECT_EQ(3, bandit.optimal_arm_probabilities().size());
    
    EXPECT_GT(bandit.optimal_arm_probabilities()[0], .15);
    EXPECT_LT(bandit.optimal_arm_probabilities()[0], .25);

    EXPECT_GT(bandit.optimal_arm_probabilities()[1], .7);
    EXPECT_LT(bandit.optimal_arm_probabilities()[1], .8);
    
    EXPECT_GT(bandit.optimal_arm_probabilities()[2], .01);
    EXPECT_LT(bandit.optimal_arm_probabilities()[2], .04);

    const Vector &value(bandit.value_remaining_distribution());
    EXPECT_EQ(value.size(), ndraws);
  }

}  // namespace

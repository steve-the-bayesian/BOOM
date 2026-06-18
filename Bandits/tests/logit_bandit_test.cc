#include "gtest/gtest.h"

#include "Bandits/LogitBandit.hpp"
#include "Bandits/LinearBanditEncoder.hpp"

#include "Models/Glm/BinomialLogitModel.hpp"
#include "Models/MvnModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialLogitAuxmixSampler.hpp"

#include "distributions.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class LogitBanditTest : public ::testing::Test {
   protected:
    LogitBanditTest() {
      GlobalRng::rng.seed(8675309);

      xp_.add_factor("ButtonPosition", {"Left", "Right"});
      xp_.add_factor("ButtonColor", {"Red", "Blue"});

      arm_map_.reset(new ArmMap(xp_));
      position_encoder_.reset(new ExperimentArmEncoder("ButtonPosition", arm_map_));
      color_encoder_.reset(new ExperimentArmEncoder("ButtonColor", arm_map_));

      dataset_encoder_.reset(new DatasetEncoder);
      dataset_encoder_->add_encoder(position_encoder_);
      dataset_encoder_->add_encoder(color_encoder_);

      encoder_.reset(new LinearBanditEncoder(arm_map_, dataset_encoder_));

      // intercept + ButtonPosition dim + ButtonColor dim
      int xdim = 1 + position_encoder_->dim() + color_encoder_->dim();
      model_.reset(new BinomialLogitModel(xdim));

      NEW(MvnModel, prior)(xdim, 0.0, 1.0);
      NEW(BinomialLogitAuxmixSampler, sampler)(model_.get(), prior);
      model_->set_method(sampler);

      bandit_.reset(new LogitBandit(model_, encoder_));
    }

    ExperimentStructure xp_;
    Ptr<ArmMap> arm_map_;
    Ptr<ExperimentArmEncoder> position_encoder_;
    Ptr<ExperimentArmEncoder> color_encoder_;
    Ptr<DatasetEncoder> dataset_encoder_;
    Ptr<LinearBanditEncoder> encoder_;
    Ptr<BinomialLogitModel> model_;
    Ptr<LogitBandit> bandit_;
  };

  // Fixture that adds a numeric context variable (x1) to the encoder.
  // This exercises the code paths that handle non-empty context.
  class LogitBanditWithContextTest : public ::testing::Test {
   protected:
    LogitBanditWithContextTest() {
      GlobalRng::rng.seed(8675309);

      xp_.add_factor("ButtonPosition", {"Left", "Right"});
      xp_.add_factor("ButtonColor", {"Red", "Blue"});

      arm_map_.reset(new ArmMap(xp_));
      position_encoder_.reset(new ExperimentArmEncoder("ButtonPosition", arm_map_));
      color_encoder_.reset(new ExperimentArmEncoder("ButtonColor", arm_map_));
      x1_encoder_.reset(new IdentityEncoder("x1"));

      dataset_encoder_.reset(new DatasetEncoder);
      dataset_encoder_->add_encoder(position_encoder_);
      dataset_encoder_->add_encoder(color_encoder_);
      dataset_encoder_->add_encoder(x1_encoder_);

      encoder_.reset(new LinearBanditEncoder(arm_map_, dataset_encoder_));

      // intercept + ButtonPosition dim + ButtonColor dim + x1 dim
      int xdim = 1 + position_encoder_->dim() + color_encoder_->dim() + x1_encoder_->dim();
      model_.reset(new BinomialLogitModel(xdim));

      NEW(MvnModel, prior)(xdim, 0.0, 1.0);
      NEW(BinomialLogitAuxmixSampler, sampler)(model_.get(), prior);
      model_->set_method(sampler);

      bandit_.reset(new LogitBandit(model_, encoder_));
    }

    Ptr<MixedMultivariateData> make_context(double x1_value) {
      DataTable table;
      Vector v(1);
      v[0] = x1_value;
      table.append_variable(v, "x1");
      return table.row(0);
    }

    ExperimentStructure xp_;
    Ptr<ArmMap> arm_map_;
    Ptr<ExperimentArmEncoder> position_encoder_;
    Ptr<ExperimentArmEncoder> color_encoder_;
    Ptr<IdentityEncoder> x1_encoder_;
    Ptr<DatasetEncoder> dataset_encoder_;
    Ptr<LinearBanditEncoder> encoder_;
    Ptr<BinomialLogitModel> model_;
    Ptr<LogitBandit> bandit_;
  };

  TEST_F(LogitBanditTest, SmokeTest) {}

  TEST_F(LogitBanditTest, NumberOfArmsTest) {
    // 2 positions x 2 colors = 4 arms
    EXPECT_EQ(4, bandit_->number_of_arms());
  }

  TEST_F(LogitBanditTest, ObserveDataTest) {
    MixedMultivariateData ctx;
    bandit_->observe_data(0, 3, 5, ctx);
    bandit_->observe_data(1, 1, 5, ctx);
    bandit_->observe_data(2, 2, 5, ctx);
    EXPECT_EQ(3, model_->dat().size());
  }

  TEST_F(LogitBanditTest, UpdatePosteriorSansContextTest) {
    MixedMultivariateData ctx;
    bandit_->observe_data(0, 5, 10, ctx);
    bandit_->observe_data(1, 2, 10, ctx);

    EXPECT_EQ(0, bandit_->ndraws());
    bandit_->update_posterior(200);
    EXPECT_EQ(200, bandit_->ndraws());
  }

  TEST_F(LogitBanditWithContextTest, UpdatePosteriorWithContextTest) {
    Ptr<MixedMultivariateData> ctx = make_context(1.5);
    bandit_->observe_data(0, 5, 10, *ctx);
    bandit_->observe_data(1, 2, 10, *ctx);

    EXPECT_EQ(0, bandit_->ndraws());
    bandit_->update_posterior(200);
    EXPECT_EQ(200, bandit_->ndraws());
  }

  // Arm index 0 corresponds to (ButtonPosition=Left, ButtonColor=Red).
  // Feed it many successes so the posterior should favor arm 0, even when
  // a numeric context variable is present.
  TEST_F(LogitBanditWithContextTest, OptimalArmProbWithContextTest) {
    Ptr<MixedMultivariateData> ctx = make_context(1.5);
    bandit_->observe_data(0, 90, 100, *ctx);
    bandit_->observe_data(1, 30, 100, *ctx);
    bandit_->observe_data(2, 20, 100, *ctx);
    bandit_->observe_data(3, 10, 100, *ctx);

    bandit_->update_posterior(1000);

    Vector probs = bandit_->optimal_arm_probabilities(*ctx);
    EXPECT_EQ(bandit_->number_of_arms(), probs.size());

    // Probabilities sum to 1.
    EXPECT_NEAR(1.0, probs.sum(), 1e-10);

    // Arm 0 dominates.
    for (int i = 1; i < bandit_->number_of_arms(); ++i) {
      EXPECT_GT(probs[0], probs[i]);
    }
  }
  
  TEST_F(LogitBanditTest, ArmPredictorsTest) {
    MixedMultivariateData ctx;
    Matrix pred = bandit_->arm_predictors(ctx);

    EXPECT_EQ(bandit_->number_of_arms(), pred.nrow());
    // intercept(1) + ButtonPosition(1) + ButtonColor(1) = 3
    EXPECT_EQ(1 + position_encoder_->dim() + color_encoder_->dim(), pred.ncol());

    // Each row starts with an intercept of 1.
    for (int i = 0; i < pred.nrow(); ++i) {
      EXPECT_DOUBLE_EQ(1.0, pred(i, 0));
    }

    // All rows should be distinct (each arm has a unique predictor).
    for (int i = 0; i < pred.nrow(); ++i) {
      for (int j = i + 1; j < pred.nrow(); ++j) {
        EXPECT_FALSE(VectorEquals(pred.row(i), pred.row(j)));
      }
    }
  }

  // Arm index 0 corresponds to (ButtonPosition=Left, ButtonColor=Red).
  // Feed it many successes so the posterior should favor arm 0.
  TEST_F(LogitBanditTest, OptimalArmProbTest) {
    MixedMultivariateData ctx;
    bandit_->observe_data(0, 90, 100, ctx);
    bandit_->observe_data(1, 30, 100, ctx);
    bandit_->observe_data(2, 20, 100, ctx);
    bandit_->observe_data(3, 10, 100, ctx);

    bandit_->update_posterior(1000);

    Vector probs = bandit_->optimal_arm_probabilities(ctx);
    EXPECT_EQ(bandit_->number_of_arms(), probs.size());

    // Probabilities sum to 1.
    EXPECT_NEAR(1.0, probs.sum(), 1e-10);

    // Arm 0 dominates.
    for (int i = 1; i < bandit_->number_of_arms(); ++i) {
      EXPECT_GT(probs[0], probs[i]);
    }
  }

  TEST_F(LogitBanditTest, ValueTest) {
    MixedMultivariateData ctx;
    bandit_->observe_data(0, 90, 100, ctx);
    bandit_->observe_data(1, 30, 100, ctx);
    bandit_->observe_data(2, 20, 100, ctx);
    bandit_->observe_data(3, 10, 100, ctx);

    bandit_->update_posterior(500);

    // After seeing data where arm 0 dominates, the model's prediction
    // for arm 0 should exceed predictions for the other arms.
    double v0 = bandit_->value(0, ctx);
    for (int i = 1; i < bandit_->number_of_arms(); ++i) {
      EXPECT_GT(v0, bandit_->value(i, ctx));
    }
  }

  // Arm 0 has overwhelming success.  After training, thompson() should return
  // arm 0's labels far more often than any other arm.
  TEST_F(LogitBanditTest, ThompsonTest) {
    MixedMultivariateData ctx;
    bandit_->observe_data(0, 90, 100, ctx);
    bandit_->observe_data(1, 10, 100, ctx);
    bandit_->observe_data(2, 10, 100, ctx);
    bandit_->observe_data(3, 10, 100, ctx);

    bandit_->update_posterior(500);

    const std::vector<std::string> arm0_labels = encoder_->arm_values(0);
    int arm0_count = 0;
    const int nsamples = 200;
    for (int i = 0; i < nsamples; ++i) {
      std::vector<std::string> chosen = bandit_->thompson(ctx);
      ASSERT_EQ(arm0_labels.size(), chosen.size());
      if (chosen == arm0_labels) {
        ++arm0_count;
      }
    }
    // With arm 0 strongly dominant, the vast majority of samples should pick it.
    EXPECT_GT(arm0_count, nsamples / 2);
  }

  // thompson() with a context variable present.
  TEST_F(LogitBanditWithContextTest, ThompsonWithContextTest) {
    Ptr<MixedMultivariateData> ctx = make_context(1.5);
    bandit_->observe_data(0, 90, 100, *ctx);
    bandit_->observe_data(1, 10, 100, *ctx);
    bandit_->observe_data(2, 10, 100, *ctx);
    bandit_->observe_data(3, 10, 100, *ctx);

    bandit_->update_posterior(500);

    const std::vector<std::string> arm0_labels = encoder_->arm_values(0);
    int arm0_count = 0;
    const int nsamples = 200;
    for (int i = 0; i < nsamples; ++i) {
      std::vector<std::string> chosen = bandit_->thompson(*ctx);
      ASSERT_EQ(arm0_labels.size(), chosen.size());
      if (chosen == arm0_labels) {
        ++arm0_count;
      }
    }
    EXPECT_GT(arm0_count, nsamples / 2);
  }

  // thompson() must return a valid arm label vector even when ndraws == 1.
  TEST_F(LogitBanditTest, ThompsonSingleDrawTest) {
    MixedMultivariateData ctx;
    bandit_->observe_data(0, 5, 10, ctx);
    bandit_->update_posterior(1);
    ASSERT_EQ(1, bandit_->ndraws());

    std::vector<std::string> chosen = bandit_->thompson(ctx);
    // The result should have one label per experiment factor.
    EXPECT_EQ(2u, chosen.size());
    // Labels must be non-empty strings.
    for (const auto &label : chosen) {
      EXPECT_FALSE(label.empty());
    }
  }

}  // namespace

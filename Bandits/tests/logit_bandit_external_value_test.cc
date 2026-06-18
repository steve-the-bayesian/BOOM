#include "gtest/gtest.h"

#include "Bandits/LogitBanditExternalValue.hpp"
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

  class LogitBanditExternalValueTest : public ::testing::Test {
   protected:
    LogitBanditExternalValueTest() {
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

      int xdim = 1 + position_encoder_->dim() + color_encoder_->dim();
      model_.reset(new BinomialLogitModel(xdim));

      NEW(MvnModel, prior)(xdim, 0.0, 1.0);
      NEW(BinomialLogitAuxmixSampler, sampler)(model_.get(), prior);
      model_->set_method(sampler);

      // Value function: identity on the success probability (ignores arm labels).
      auto value_fn = [](double prob, const std::vector<std::string> &) {
        return prob;
      };

      bandit_.reset(new LogitBanditExternalValue(model_, encoder_, value_fn));
    }

    ExperimentStructure xp_;
    Ptr<ArmMap> arm_map_;
    Ptr<ExperimentArmEncoder> position_encoder_;
    Ptr<ExperimentArmEncoder> color_encoder_;
    Ptr<DatasetEncoder> dataset_encoder_;
    Ptr<LinearBanditEncoder> encoder_;
    Ptr<BinomialLogitModel> model_;
    Ptr<LogitBanditExternalValue> bandit_;
  };

  TEST_F(LogitBanditExternalValueTest, SmokeTest) {}

  // Arm 0 has overwhelming success.  With an identity value function,
  // thompson() should pick arm 0 far more than half the time.
  TEST_F(LogitBanditExternalValueTest, ThompsonTest) {
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
    EXPECT_GT(arm0_count, nsamples / 2);
  }

  // Verify that a value function which inverts the ranking (returns 1 - prob)
  // causes thompson() to consistently avoid arm 0 even though it has the
  // highest success probability.
  TEST_F(LogitBanditExternalValueTest, ThompsonCustomValueFunctionTest) {
    auto inverted_value_fn = [](double prob, const std::vector<std::string> &) {
      return 1.0 - prob;
    };

    ExperimentStructure xp;
    xp.add_factor("ButtonPosition", {"Left", "Right"});
    xp.add_factor("ButtonColor", {"Red", "Blue"});

    Ptr<ArmMap> arm_map(new ArmMap(xp));
    Ptr<ExperimentArmEncoder> pos_enc(new ExperimentArmEncoder("ButtonPosition", arm_map));
    Ptr<ExperimentArmEncoder> col_enc(new ExperimentArmEncoder("ButtonColor", arm_map));
    Ptr<DatasetEncoder> ds_enc(new DatasetEncoder);
    ds_enc->add_encoder(pos_enc);
    ds_enc->add_encoder(col_enc);
    Ptr<LinearBanditEncoder> enc(new LinearBanditEncoder(arm_map, ds_enc));

    int xdim = 1 + pos_enc->dim() + col_enc->dim();
    Ptr<BinomialLogitModel> mdl(new BinomialLogitModel(xdim));
    NEW(MvnModel, prior)(xdim, 0.0, 1.0);
    NEW(BinomialLogitAuxmixSampler, sampler)(mdl.get(), prior);
    mdl->set_method(sampler);

    Ptr<LogitBanditExternalValue> inv_bandit(
        new LogitBanditExternalValue(mdl, enc, inverted_value_fn));

    MixedMultivariateData ctx;
    inv_bandit->observe_data(0, 90, 100, ctx);
    inv_bandit->observe_data(1, 10, 100, ctx);
    inv_bandit->observe_data(2, 10, 100, ctx);
    inv_bandit->observe_data(3, 10, 100, ctx);
    inv_bandit->update_posterior(500);

    // With inverted value, arm 0 (the best success arm) should be chosen rarely.
    const std::vector<std::string> arm0_labels = enc->arm_values(0);
    int arm0_count = 0;
    const int nsamples = 200;
    for (int i = 0; i < nsamples; ++i) {
      std::vector<std::string> chosen = inv_bandit->thompson(ctx);
      ASSERT_EQ(arm0_labels.size(), chosen.size());
      if (chosen == arm0_labels) {
        ++arm0_count;
      }
    }
    EXPECT_LT(arm0_count, nsamples / 2);
  }

  // thompson() must return a valid label vector even with a single posterior draw.
  TEST_F(LogitBanditExternalValueTest, ThompsonSingleDrawTest) {
    MixedMultivariateData ctx;
    bandit_->observe_data(0, 5, 10, ctx);
    bandit_->update_posterior(1);
    ASSERT_EQ(1, bandit_->ndraws());

    std::vector<std::string> chosen = bandit_->thompson(ctx);
    EXPECT_EQ(2u, chosen.size());
    for (const auto &label : chosen) {
      EXPECT_FALSE(label.empty());
    }
  }

}  // namespace

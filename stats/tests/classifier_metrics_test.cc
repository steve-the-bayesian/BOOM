#include "gtest/gtest.h"

#include "stats/classifier_metrics.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class TjurTest : public ::testing::Test {
   protected:
    TjurTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(TjurTest, TestPerfect) {
    std::vector<bool> truth;
    Vector pred;
    int n = 100;
    for (int i = 0; i < n; ++i) {
      double u = runif_mt(GlobalRng::rng, 0, 1);
      bool success = u > .5;
      truth.push_back(success);
      if (success) {
        pred.push_back(1.0);
      } else {
        pred.push_back(0.0);
      }
    }
    double R2 = TjurR2(truth, pred);
    EXPECT_GT(R2, .99);
    EXPECT_LE(R2, 1.0);
  }

  TEST_F(TjurTest, TestPerfectButImbalanced) {
    std::vector<bool> truth;
    Vector pred;
    int n = 100;
    for (int i = 0; i < n; ++i) {
      double u = runif_mt(GlobalRng::rng, 0, 1);
      bool success = u > .9;
      truth.push_back(success);
      if (success) {
        pred.push_back(1.0);
      } else {
        pred.push_back(0.0);
      }
    }
    double R2 = TjurR2(truth, pred);
    EXPECT_GT(R2, .99);
    EXPECT_LE(R2, 1.0);
  }
  
  // Test the Tjur R2 when the predicted values have nothing to do with the
  // success/failure outcomes.  Successes are imbalanced.
  TEST_F(TjurTest, TestRandomNoise) {
    std::vector<bool> truth;
    Vector pred;
    Int n = 100000;
    for (Int i = 0; i < n; ++i) {
      double u = runif_mt(GlobalRng::rng, 0, 1);
      bool success = u > .8;
      truth.push_back(success);
      pred.push_back(runif_mt(GlobalRng::rng, 0, 1));
    }
    double R2 = TjurR2(truth, pred);
    EXPECT_LT(R2, .02);
    EXPECT_GE(R2, 0.0);
  }

}  // namespace

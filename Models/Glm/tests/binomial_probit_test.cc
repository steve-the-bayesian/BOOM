#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/BinomialProbitModel.hpp"
#include "Models/Glm/PosteriorSamplers/BinomialProbitDataImputer.hpp"

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
  
}  // namespace

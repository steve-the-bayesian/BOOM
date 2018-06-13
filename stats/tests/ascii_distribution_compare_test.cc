#include "gtest/gtest.h"

#include "distributions.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class AsciiDistributionCompareTest : public ::testing::Test {
   protected:
    AsciiDistributionCompareTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // This test has to be judged by eye.  Change EXPECT_TRUE(true) to
  // EXPECT_TRUE(false) to view the plots.
  
  TEST_F(AsciiDistributionCompareTest, NormalDistributionLargeSample) {
    Vector draws(10000);
    for (int i = 0; i < draws.size(); ++i) {
      draws[i] = rnorm(3, 7);
    }
    AsciiDistributionCompare plot(draws, 3.0);
    EXPECT_TRUE(true) << plot;
  }

  TEST_F(AsciiDistributionCompareTest, NormalDistributionTwoSample) {
    Vector x(10000);
    Vector y(1000);
    for (int i = 0; i < x.size(); ++i) {
      x[i] = rnorm(3, 7);
    }
    for (int i = 0; i < y.size(); ++i) {
      y[i] = rnorm(4, 3);
    }

    AsciiDistributionCompare plot(x, y);
    EXPECT_TRUE(true) << plot;
  }
  
}  // namespace

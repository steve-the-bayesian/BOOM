#include "gtest/gtest.h"

#include "stats/summary.hpp"
#include "distributions.hpp"
#include "LinAlg/Vector.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class SummaryTest : public ::testing::Test {
   protected:
    SummaryTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SummaryTest, NumericTest) {
    int sample_size = 10000;
    Vector draws = rnorm_vector(sample_size, 3, 7);
    NumericSummary summary(draws);
    EXPECT_EQ(summary.sample_size(), sample_size);

  }


}  // namespace

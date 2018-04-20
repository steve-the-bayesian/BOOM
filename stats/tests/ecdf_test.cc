#include "gtest/gtest.h"
#include "stats/ECDF.hpp"

#include "LinAlg/Vector.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class EcdfTest : public ::testing::Test {
   protected:
    EcdfTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // Check that a well populated empirical CDF based on normal data matches the
  // theoretical normal CDF.
  TEST_F(EcdfTest, NormalUse) {
    int n = 100000;
    Vector values(n);
    for (int i = 0; i < n; ++i) {
      values[i] = rnorm();
    }

    ECDF ecdf(values);
    EXPECT_NEAR(ecdf(-2), pnorm(-2), .01);
    EXPECT_NEAR(ecdf(1.6), pnorm(1.6), .01);
    EXPECT_NEAR(ecdf(3.1), pnorm(3.1), .01);

    EXPECT_NEAR(ecdf.quantile(.025), qnorm(.025), .01);
    EXPECT_NEAR(ecdf.quantile(.975), qnorm(.975), .01);
  }

}  // namespace

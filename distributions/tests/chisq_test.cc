#include "gtest/gtest.h"
#include "distributions.hpp"
#include <iostream>

namespace {
  using namespace BOOM;
  using std::cout;
  using std::endl;


  TEST(ChisqTest, Quantiles) {
    // Check that results are good to high degrees of freedom.
    for (int i = 1; i < 100000; ++i) {
      EXPECT_FALSE(std::isnan(qchisq(.95, i)));

      // Check that p and q are inverses.
      EXPECT_NEAR(pchisq(qchisq(.95, i), i), .95, 1e-4);
    }
  }

}  // namespace 

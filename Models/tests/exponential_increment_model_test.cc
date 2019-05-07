#include "gtest/gtest.h"
#include "Models/ExponentialIncrementModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class ExponentialIncrementTest : public ::testing::Test {
   protected:
    ExponentialIncrementTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ExponentialIncrementTest, Simulation) {
    ExponentialIncrementModel model(Vector{1.2, .3, 7});
    Vector y = model.sim();
    EXPECT_EQ(y.size(), 3 + 1);
    EXPECT_DOUBLE_EQ(0.0, y[0]);
    EXPECT_GT(y[1], 0.0);
    EXPECT_GT(y[2] - y[1], 0.0);
    EXPECT_GT(y[3] - y[2], 0.0);

    EXPECT_DOUBLE_EQ(
        model.logp(y),
        dexp(y[1], 1.2, true) +
        dexp(y[2] - y[1], .3, true) + 
        dexp(y[3] - y[2], 7, true));
  }
  
}  // namespace

#include "gtest/gtest.h"

#include "Models/GP/GaussianProcessRegressionModel.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class KernelTest : public ::testing::Test {
   protected:
    KernelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(KernelTest, RadialBasisTest) {
    RadialBasisFunction k(1.7);

    Vector x1(3);
    Vector x2(3);
    x1.randomize();
    x2.randomize();

    EXPECT_DOUBLE_EQ(k(x1, x2),
                     exp(-2 * (x1 - x2).normsq() / 1.7 / 1.7));

  }

}  // namespace

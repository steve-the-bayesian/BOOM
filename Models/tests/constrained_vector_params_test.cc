#include "gtest/gtest.h"
#include "Models/ConstrainedVectorParams.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using BOOM::uint;
  using std::endl;
  using std::cout;

  class ConstrainedVectorParamsTest : public ::testing::Test {
   protected:
    ConstrainedVectorParamsTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ConstrainedVectorParamsTest, ProportionalSumConstraintTest) {
    ConstrainedVectorParams prm(
        Vector{1.0, 2.0, 3.0},
        new ProportionalSumConstraint(2.4));
    EXPECT_DOUBLE_EQ(prm.value().sum(), 2.4);

    Vector vmin = prm.vectorize(true);
    EXPECT_EQ(vmin.size(), 2);

    prm.set(Vector{2.0, 1.0});
    EXPECT_EQ(prm.size(), 3);
    EXPECT_DOUBLE_EQ(prm.value()[0], -0.6);
    EXPECT_DOUBLE_EQ(prm.value()[1], 2.0);
    EXPECT_DOUBLE_EQ(prm.value()[2], 1.0);

  }


}  // namespace

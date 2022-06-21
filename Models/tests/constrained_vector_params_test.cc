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
    EXPECT_EQ(prm.size(false), 3);
    EXPECT_EQ(prm.size(true), 2);
    EXPECT_DOUBLE_EQ(prm.value()[0], -0.6);
    EXPECT_DOUBLE_EQ(prm.value()[1], 2.0);
    EXPECT_DOUBLE_EQ(prm.value()[2], 1.0);

    prm.set(Vector{2.0, 1.0, 3.0});
    EXPECT_EQ(prm.size(false), 3);
    EXPECT_EQ(prm.size(true), 2);
    EXPECT_DOUBLE_EQ(prm.value()[0],  2.0 * 2.4 / 6.0);
    EXPECT_DOUBLE_EQ(prm.value()[1],  1.0 * 2.4 / 6.0);
    EXPECT_DOUBLE_EQ(prm.value()[2],  3.0 * 2.4 / 6.0);
  }

  TEST_F(ConstrainedVectorParamsTest, ElementConstraintTest) {
    ConstrainedVectorParams prm(Vector{2.0, 1.0, 3.0},
                                new ElementConstraint(0, 1.0));
    EXPECT_EQ(prm.size(false), 3);
    EXPECT_EQ(prm.size(true), 2);
    EXPECT_EQ(prm.value()[0], 1.0);
    EXPECT_EQ(prm.value()[1], 1.0);
    EXPECT_EQ(prm.value()[2], 3.0);

    prm.set(Vector{5.0, 2.0, 3.8});
    EXPECT_EQ(prm.size(false), 3);
    EXPECT_EQ(prm.size(true), 2);
    EXPECT_EQ(prm.value()[0], 1.0);
    EXPECT_EQ(prm.value()[1], 2.0);
    EXPECT_EQ(prm.value()[2], 3.8);

    Vector reduced = prm.vectorize(true);
    EXPECT_EQ(reduced.size(), 2);
    EXPECT_DOUBLE_EQ(reduced[0], 2.0);
    EXPECT_DOUBLE_EQ(reduced[1], 3.8);
    prm.unvectorize(reduced);
    EXPECT_EQ(prm.size(false), 3);
    EXPECT_EQ(prm.size(true), 2);
    EXPECT_EQ(prm.value()[0], 1.0);
    EXPECT_EQ(prm.value()[1], 2.0);
    EXPECT_EQ(prm.value()[2], 3.8);
  }


}  // namespace

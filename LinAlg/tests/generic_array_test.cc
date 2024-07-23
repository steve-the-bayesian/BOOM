#include "gtest/gtest.h"
#include "LinAlg/Array.hpp"
#include "distributions.hpp"

namespace {
  using namespace BOOM;

  class GenericArrayTest : public ::testing::Test {
   protected:

    GenericArrayTest() {
      GlobalRng::rng.seed(8675309);
    }

  };

  TEST_F(GenericArrayTest, Empty) {
    GenericArray<int> empty;
  }

  TEST_F(GenericArrayTest, MiscArray) {
    std::vector<int> dims = {2, 4, 3};
    std::vector<int> values(24);
    for (int i = 0; i < values.size(); ++i) {
      values[i] = rpois(1.0);
    }
    GenericArray<int> array(dims, values);

    int val = array[{0, 0, 0}];
    EXPECT_EQ(values[0], val);
    val = array[{1, 0, 0}];
    EXPECT_EQ(values[1], val);

    EXPECT_EQ(array.ndim(), 3);
    EXPECT_EQ(array.dim(0), 2);
    EXPECT_EQ(array.dim(1), 4);
    EXPECT_EQ(array.dim(2), 3);

    val = array[{1, 3, 1}];
    EXPECT_EQ(val, values[14]);
  }


}  // namespace

#include "gtest/gtest.h"

#include "distributions.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/moments.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class MomentsTest : public ::testing::Test {
   protected:
    MomentsTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MomentsTest, TestProd) {
    std::vector<int> int_values = {8, 6, 7};
    EXPECT_EQ(8 * 6 * 7, prod(int_values));

    Vector vec(12);
    vec.randomize();
    EXPECT_EQ(prod(vec), vec.prod());
  }


}  // namespace

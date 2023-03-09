#include "gtest/gtest.h"

#include "Models/StateSpace/Filters/SparseVector.hpp"
#include "distributions.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  using std::endl;

  class SparseVectorTest : public ::testing::Test {
   protected:
    SparseVectorTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SparseVectorTest, EmptyVector) {
    SparseVector empty(3);
    EXPECT_EQ(empty.size(), 3);
  }

}  // namespace

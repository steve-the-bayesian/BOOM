#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class SelectorTest : public ::testing::Test {
   protected:
    SelectorTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SelectorTest, SparseSum) {
    Vector v(100);
    v.randomize();

    Selector inc(100);
    inc.drop_all();
    EXPECT_DOUBLE_EQ(0.0, inc.sparse_sum(v));
    
    inc.add(3);
    inc.add(17);
    inc.add(12);
    EXPECT_DOUBLE_EQ(
        inc.sparse_sum(v),
        v[3] + v[12] + v[17]);
  }

}  // namespace

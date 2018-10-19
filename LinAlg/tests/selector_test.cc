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

  TEST_F(SelectorTest, Equality) {
    Selector s1("001");
    Selector s2("101");
    EXPECT_NE(s1, s2);
    EXPECT_TRUE(s1 != s2);
    EXPECT_FALSE(s1 == s2);
    
    s2.flip(0); // now 001
    EXPECT_EQ(s1, s2);
    EXPECT_TRUE(s1 == s2);
    EXPECT_FALSE(s1 != s2);

    s2.flip(0);
    swap(s1, s2);
    Selector s3("001");
    EXPECT_EQ(s2, s3);
    EXPECT_NE(s1, s3);
    s3.flip(0);
    EXPECT_EQ(s1, s3);
    EXPECT_NE(s2, s3);
  }

  TEST_F(SelectorTest, Append) {
    Selector s1("010");
    EXPECT_EQ(s1.nvars(), 1);
    EXPECT_EQ(s1.nvars_possible(), 3);

    s1.push_back(true);
    EXPECT_EQ(s1.nvars(), 2);
    EXPECT_EQ(s1.nvars_possible(), 4);

    s1.push_back(false);
    EXPECT_EQ(s1.nvars(), 2);
    EXPECT_EQ(s1.nvars_possible(), 5);

    // Current state is "01010".
    // Set it to "0100"
    s1.erase(3);
    EXPECT_EQ(s1.nvars(), 1);
    EXPECT_EQ(s1.nvars_possible(), 4);
    EXPECT_TRUE(s1[1]);
  }

  TEST_F(SelectorTest, ToVectorTest) {
    Selector s("00010");
    Vector v = s.to_Vector();
    EXPECT_EQ(v.size(), 5);
    EXPECT_DOUBLE_EQ(v[0], 0);
    EXPECT_DOUBLE_EQ(v[1], 0);
    EXPECT_DOUBLE_EQ(v[2], 0);
    EXPECT_DOUBLE_EQ(v[3], 1);
    EXPECT_DOUBLE_EQ(v[4], 0);
  }
  
  TEST_F(SelectorTest, SelectRowsTest) {
    Matrix big(10, 4);
    big.randomize();

    Selector inc(10, false);
    inc.add(2);
    inc.add(7);

    Matrix small = inc.select_rows(big);
    EXPECT_EQ(2, small.nrow());
    EXPECT_EQ(4, small.ncol());
    EXPECT_TRUE(VectorEquals(small.row(0), big.row(2)));

    big.randomize();
    ConstSubMatrix big_view(big);
    small = inc.select_rows(big_view);
    EXPECT_EQ(2, small.nrow());
    EXPECT_EQ(4, small.ncol());
    EXPECT_TRUE(VectorEquals(small.row(0), big.row(2)));

    big.randomize();
    SubMatrix mutable_big_view(big);
    small = inc.select_rows(big_view);
    EXPECT_EQ(2, small.nrow());
    EXPECT_EQ(4, small.ncol());
    EXPECT_TRUE(VectorEquals(small.row(0), big.row(2)));
  }
  
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

  // This test checks the Selector's ability to select from std::vector.
  TEST_F(SelectorTest, VectorInt) {
    std::vector<int> big = {1, 2, 3, 4, 5};
    Selector inc("10010");
    std::vector<int> small = inc.select(big);
    EXPECT_EQ(2, small.size());
    EXPECT_EQ(1, small[0]);
    EXPECT_EQ(4, small[1]);
  }

  TEST_F(SelectorTest, SelectorMatrixTest) {
    Selector inc_vec("111101");
    SelectorMatrix inc(3, 2, inc_vec);
    EXPECT_EQ(3, inc.nrow());
    EXPECT_EQ(2, inc.ncol());

    // Check that the input vector fills column-by-column.
    EXPECT_TRUE(inc(0, 0));
    EXPECT_TRUE(inc(1, 0));
    EXPECT_TRUE(inc(2, 0));
    EXPECT_TRUE(inc(0, 1));
    EXPECT_FALSE(inc(1, 1));
    EXPECT_TRUE(inc(2, 1));

    EXPECT_EQ(inc_vec, inc.vectorize());
    
    inc.drop(0, 1);
    EXPECT_FALSE(inc(0, 1));
    inc.add(0, 1);
    EXPECT_TRUE(inc(0, 1));
    inc.flip(0, 1);
    EXPECT_FALSE(inc(0, 1));
    inc.flip(0, 1);
    EXPECT_TRUE(inc(0, 1));

    EXPECT_FALSE(inc.all_in());
    EXPECT_FALSE(inc.all_out());
    inc.drop_all();
    for (int i = 0; i < inc.nrow(); ++i) {
      for (int j = 0; j < inc.ncol(); ++j) {
        EXPECT_FALSE(inc(i, j));
      }
    }
    EXPECT_TRUE(inc.all_out());

    inc.add_all();
    for (int i = 0; i < inc.nrow(); ++i) {
      for (int j = 0; j < inc.ncol(); ++j) {
        EXPECT_TRUE(inc(i, j));
      }
    }
    EXPECT_TRUE(inc.all_in());

    SelectorMatrix wide(4, 3, Selector("100000010010"));
    // 1 0 0
    // 0 0 0
    // 0 0 1
    // 0 1 0
    Selector any(wide.row_or());
    EXPECT_TRUE(any[0]);
    EXPECT_FALSE(any[1]);
    EXPECT_TRUE(any[2]);
    EXPECT_TRUE(any[3]);
  }
  
}  // namespace

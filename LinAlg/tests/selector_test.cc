#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
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

  TEST_F(SelectorTest, Complement) {
    Selector s1("101001101");
    Selector s2 = s1.complement();
    EXPECT_EQ(s2.nvars_possible(), s1.nvars_possible());
    for (int i = 0; i < s1.nvars_possible(); ++i) {
      EXPECT_EQ(s1[i], !s2[i]);
    }
  }


  TEST_F(SelectorTest, Indexing) {
    Selector s1("101001101");
    EXPECT_EQ(s1.nvars(), 5);
    EXPECT_EQ(s1.nvars_possible(), 9);
    EXPECT_EQ(s1.nvars_excluded(), 4);

    EXPECT_EQ(0, s1.indx(0));
    EXPECT_EQ(2, s1.indx(1));
    EXPECT_EQ(5, s1.indx(2));
    EXPECT_EQ(6, s1.indx(3));
    EXPECT_EQ(8, s1.indx(4));

    EXPECT_EQ(0, s1.INDX(0));
    EXPECT_EQ(1, s1.INDX(2));
    EXPECT_EQ(2, s1.INDX(5));
    EXPECT_EQ(3, s1.INDX(6));
    EXPECT_EQ(4, s1.INDX(8));
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

    std::vector<int> expanded = inc.expand(small);
    EXPECT_EQ(expanded.size(), 5);
    EXPECT_EQ(expanded[0], big[0]);
    EXPECT_EQ(expanded[1], 0);
    EXPECT_EQ(expanded[2], 0);
    EXPECT_EQ(expanded[3], big[3]);
    EXPECT_EQ(expanded[4], 0);
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
    Selector any(wide.row_any());
    EXPECT_TRUE(any[0]);
    EXPECT_FALSE(any[1]);
    EXPECT_TRUE(any[2]);
    EXPECT_TRUE(any[3]);

    Selector all(wide.row_all());
    EXPECT_FALSE(all[0]);
    EXPECT_FALSE(all[0]);
    EXPECT_FALSE(all[0]);
    EXPECT_FALSE(all[3]);

    Selector row2 = wide.row(2);
    EXPECT_EQ(3, row2.nvars_possible());
    EXPECT_FALSE(row2[0]);
    EXPECT_FALSE(row2[1]);
    EXPECT_TRUE(row2[2]);

    Matrix selectable(4, 3);
    selectable.randomize();
    Vector selected = wide.vector_select(selectable);
    EXPECT_EQ(3, selected.size());
    EXPECT_DOUBLE_EQ(selected[0], selectable(0, 0));
    EXPECT_DOUBLE_EQ(selected[1], selectable(3, 1));
    EXPECT_DOUBLE_EQ(selected[2], selectable(2, 2));

    Matrix expanded(4, 3, 0.0);
    expanded(0, 0) = selectable(0, 0);
    expanded(3, 1) = selectable(3, 1);
    expanded(2, 2) = selectable(2, 2);

    wide.add(3, 0);
    wide.add(3, 2);
    all = wide.row_all();
    EXPECT_FALSE(all[0]);
    EXPECT_FALSE(all[0]);
    EXPECT_FALSE(all[0]);
    EXPECT_TRUE(all[3]);
  }

  TEST_F(SelectorTest, EmptySelectorTest) {
    Selector empty(4, false);
    EXPECT_EQ(0, empty.nvars());
    EXPECT_EQ(4, empty.nvars_possible());
  }

  TEST_F(SelectorTest, DiagonalMatrixTest) {
    Selector empty(4, false);
    Selector full(4, true);
    Selector one(4, false);
    one.add(2);
    Selector three(4, true);
    three.drop(2);

    Vector v(4);
    v.randomize();
    DiagonalMatrix dmat(v);

    EXPECT_EQ(empty.select_square(dmat).nrow(), 0);
    EXPECT_EQ(empty.select_square(dmat).ncol(), 0);
    EXPECT_EQ(full.select_square(dmat).nrow(), 4);
    EXPECT_EQ(full.select_square(dmat).ncol(), 4);
    EXPECT_TRUE(VectorEquals(full.select_square(dmat).diag(), v));

    EXPECT_EQ(one.select_square(dmat).nrow(), 1);
    EXPECT_EQ(one.select_square(dmat).ncol(), 1);
    EXPECT_TRUE(VectorEquals( one.select_square(dmat).diag(), one.select(v)));

    EXPECT_EQ(three.select_square(dmat).nrow(), 3);
    EXPECT_EQ(three.select_square(dmat).ncol(), 3);
    EXPECT_TRUE(VectorEquals(three.select_square(dmat).diag(),
                             three.select(v)));
  }

  TEST_F(SelectorTest, FillMissingValues) {
    Vector x(5);
    x.randomize();

    Selector all(5, true);
    Selector none(5, false);

    EXPECT_TRUE(VectorEquals(x, all.fill_missing_elements(x, 3.0)));
    EXPECT_TRUE(VectorEquals(Vector(5, 3.0),
                             none.fill_missing_elements(x, 3.0)));

    x.randomize();
    Selector three("11010");
    Vector y = x;
    three.fill_missing_elements(x, 3.0);
    y[2] = 3.0;
    y[4] = 3.0;
    EXPECT_TRUE(VectorEquals(x, y));

    Vector values = {1.2, 2.4};
    three.fill_missing_elements(x, values);
    y[2] = 1.2;
    y[4] = 2.4;
    EXPECT_TRUE(VectorEquals(x, y));
  }

}  // namespace

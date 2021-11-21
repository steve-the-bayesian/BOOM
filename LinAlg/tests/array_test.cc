#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Array.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class ArrayTest : public ::testing::Test {
   protected:
    ArrayTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(ArrayTest, Array) {
    Array empty;
    EXPECT_EQ(0, empty.ndim());

    std::vector<int> dim = {2, 4, 3};
    Array array(dim, 0.0);
    EXPECT_EQ(array.ndim(), 3);
    EXPECT_EQ(array.dim()[0], dim[0]);
    EXPECT_EQ(array.dim()[1], dim[1]);
    EXPECT_EQ(array.dim()[2], dim[2]);
    EXPECT_DOUBLE_EQ(array(1, 1, 0), 0.0);

    EXPECT_EQ(array.size(), 2 * 4 * 3);

    Matrix foo(2, 3);
    foo.randomize();
    array.slice(-1, 2, -1) = foo;
    for (int i = 0; i < foo.nrow(); ++i) {
      for (int j = 0; j < foo.ncol(); ++j) {
        EXPECT_DOUBLE_EQ(array(i, 2, j), foo(i, j));
      }
    }

    array(0, 0, 0) = 1;
    array(0, 0, 1) = 2;
    array(0, 0, 2) = 3;
    array(0, 1, 0) = 4;
    array(0, 1, 1) = 5;

    // Check that the array is printed correctly.
    std::string array_string = array.to_string();
    EXPECT_EQ(array_string.substr(0, 10),
              "1 2 3 \n4 5");
  }

  // TODO
  TEST_F(ArrayTest, ArrayViewTest) {
    std::vector<int> dim = {2, 4, 3};
    Array array(dim, 0.0);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int k = 0; k < 3; ++k) {
          array(i, j, k) = rnorm();
        }
      }
    }
    EXPECT_EQ(array.slice(0, -1, -1).ndim(), 2);
    EXPECT_EQ(array.slice(-1, 0, -1).ndim(), 2);
    EXPECT_EQ(array.slice(-1, -1, 0).ndim(), 2);
  }

  // TODO
  TEST_F(ArrayTest, ConstArrayViewTest) { }

  TEST_F(ArrayTest, IterationOrder) {
    Array arr(std::vector<int>{3, 4, 5});
    arr.randomize();

    auto it = arr.abegin();
    EXPECT_EQ(it.position(), (std::vector<int>{0, 0, 0}));
    ++it;
    EXPECT_EQ(it.position(), (std::vector<int>{1, 0, 0}));
    ++it;
    EXPECT_EQ(it.position(), (std::vector<int>{2, 0, 0}));
    ++it;
    EXPECT_EQ(it.position(), (std::vector<int>{0, 1, 0}));
    EXPECT_DOUBLE_EQ(*it, arr[it.position()]);
  }

}  // namespace

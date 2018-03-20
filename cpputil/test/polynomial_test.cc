#include "gtest/gtest.h"
#include "cpputil/Polynomial.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
  TEST(polynomial, works_as_intended) {
    Polynomial p1({-5., 4});
    Polynomial p2({-6., 3, 2});

    EXPECT_DOUBLE_EQ(p1(3.2), 4 * 3.2 - 5);
    EXPECT_DOUBLE_EQ(p2(3.2), -6  + 3 * 3.2 + 2 * 3.2 * 3.2);

    Polynomial p3 = p1 * p2;
    EXPECT_TRUE(VectorEquals(p3.coefficients(), Vector{30, -39, 2, 8}));

    Polynomial p4 = p1 + p2;
    EXPECT_TRUE(VectorEquals(p4.coefficients(),
                             Vector({-11, 7, 2})));

    Polynomial p5 = p2 - p1;
    EXPECT_TRUE(VectorEquals(p5.coefficients(),
                             Vector({-1, -1, 2})));
    
  }
  
}  // namespace

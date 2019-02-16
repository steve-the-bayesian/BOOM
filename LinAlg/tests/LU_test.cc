#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/LU.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class LuTest : public ::testing::Test {
   protected:
    LuTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(LuTest, Decomposition) {
    Matrix A(3, 3);
    A.randomize();
    LU lu(A);
    EXPECT_EQ(3, lu.nrow());
    EXPECT_EQ(3, lu.ncol());

    Matrix Id(3, 3, 0.0);
    Id.diag() += 1.0;
    Matrix Ainv = lu.solve(Id);
    Matrix Ainv2 = A.inv();
    EXPECT_TRUE(MatrixEquals(Ainv, Ainv2));

    lu.clear();
    EXPECT_EQ(0, lu.nrow());
    EXPECT_EQ(0, lu.ncol());

    lu.decompose(A);
    EXPECT_EQ(3, lu.nrow());
    EXPECT_EQ(3, lu.ncol());
    
    Vector v(A.ncol());
    v.randomize();
    EXPECT_TRUE(VectorEquals(A.solve(v), lu.solve(v)));
  }

  TEST_F(LuTest, Determinant) {
    Matrix A(2, 2);
    A.randomize();
    double a = A(0, 0);
    double b = A(0, 1);
    double c = A(1, 0);
    double d = A(1, 1);
    LU lu(A);
    EXPECT_NEAR(lu.det(), a * d - b * c, 1e-6);

    Matrix B("4 1 | 1 5");
    lu.decompose(B);
    EXPECT_NEAR(std::log(lu.det()), lu.logdet(), 1e-7);

    Matrix C("1 2 | 3 4");  // det is 4 - 6 = -2
    lu.decompose(C);
    EXPECT_DOUBLE_EQ(lu.logdet(), negative_infinity());
  }
  
}  // namespace

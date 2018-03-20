#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/QR.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class QrTest : public ::testing::Test {
   protected:
    QrTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(QrTest, Decomposition) {
    Matrix A(10, 3);
    A.randomize();
    QR qr(A);

    Matrix R = qr.getR();
    EXPECT_EQ(3, R.nrow());
    EXPECT_EQ(3, R.ncol());
    // R should be upper triangular.
    for (int i = 1; i < 3; ++i) {
      for (int j = 0; j < i; ++j) {
        EXPECT_DOUBLE_EQ(0.0, R(i, j));
      }
    }

    Matrix Q = qr.getQ();
    EXPECT_EQ(10, Q.nrow());
    EXPECT_EQ(3, Q.ncol());

    Matrix B = Q * R;
    EXPECT_TRUE(MatrixEquals(A, B))
        << "A = " << endl << A << endl
        << "B = " << endl << B << endl;

    EXPECT_TRUE(MatrixEquals(Q.Tmult(Q), SpdMatrix(3, 1.0)));
  }
  
}  // namespace

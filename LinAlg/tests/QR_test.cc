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

  TEST_F(QrTest, Determinant) {
    Matrix A(2, 2);
    A.randomize();
    double a = A(0, 0);
    double b = A(0, 1);
    double c = A(1, 0);
    double d = A(1, 1);
    QR qr(A);
    EXPECT_NEAR(qr.det(), a * d - b * c, 1e-6);
  }

  TEST_F(QrTest, Regression) {
    Matrix predictors(100, 4);
    predictors.randomize();
    predictors.col(0) = 1.0;

    Vector response(100);
    response.randomize();

    QR qr(predictors);

    // xtx.inv * xty
    // (rtr).inv * rt qty
    // r.inv rt.inv * rt * qty
    // r.inv * qty

    Matrix Q = qr.getQ();
    Vector qty = qr.Qty(response);
    Vector qty2 = Q.transpose() * response;

    Vector qr_coef = qr.Rsolve(qty);
    EXPECT_TRUE(VectorEquals(qr_coef, qr.getR().inv() * qty, 1e-5));
    
    SpdMatrix xtx = predictors.transpose() * predictors;
    Vector direct_coef = xtx.solve(predictors.transpose() * response);
    EXPECT_TRUE(VectorEquals(qr_coef, direct_coef, 1e-5));
  }

  TEST_F(QrTest, Solve) {
    Matrix A(4, 4);
    A.randomize();

    QR qr(A);
    
    // Find A.inv * B

    Matrix B(4, 2);
    B.randomize();

    Matrix QB = qr.getQ().transpose() * B;
    Matrix RinvQB = Usolve(qr.getR(), QB);
    EXPECT_TRUE(MatrixEquals(A * RinvQB, B, 1e-5))
        << "RinvQB = " << endl
        << RinvQB;
    
    Matrix AinvB = qr.solve(B);
    EXPECT_TRUE(MatrixEquals(A * AinvB, B, 1e-5));

    Vector x(4);
    x.randomize();

    Vector Ainv_x = qr.solve(x);
    EXPECT_TRUE(VectorEquals(A * Ainv_x, x, 1e-6));
    
  }
  
}  // namespace

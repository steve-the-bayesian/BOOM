#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Cholesky.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class SpdMatrixTest : public ::testing::Test {
   protected:
    SpdMatrixTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SpdMatrixTest, Constructors) {
    SpdMatrix empty;
    EXPECT_EQ(0, empty.nrow());
    EXPECT_EQ(0, empty.ncol());

    SpdMatrix single(1);
    EXPECT_EQ(1, single.nrow());
    EXPECT_EQ(1, single.ncol());
    EXPECT_DOUBLE_EQ(0.0, single(0, 0));
  }

  TEST_F(SpdMatrixTest, Multiplication) {
    SpdMatrix V(3);
    V.randomize();

    Vector v(3);
    v.randomize();
    Vector Vv = V * v;
    EXPECT_EQ(3, Vv.size());
    EXPECT_TRUE(VectorEquals(Vv, Matrix(V) * v));
    EXPECT_DOUBLE_EQ(V.row(0).dot(v), Vv[0]);
    EXPECT_DOUBLE_EQ(V.row(1).dot(v), Vv[1]);
    EXPECT_DOUBLE_EQ(V.row(2).dot(v), Vv[2]);
    
    SpdMatrix Sigma(3);
    Sigma.randomize();
    EXPECT_TRUE(Sigma.is_sym());

    Matrix VSigma = V * Sigma;
    EXPECT_TRUE(MatrixEquals(
        VSigma,
        Matrix(V) * Matrix(Sigma)));
  }

  TEST_F(SpdMatrixTest, Cholesky) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    Matrix L = Sigma.chol();
    Matrix LLT = L * L.transpose();
    EXPECT_TRUE(MatrixEquals(Sigma, LLT));

    SpdMatrix zero(4);
    bool ok = true;
    L = zero.chol(ok);
    EXPECT_FALSE(ok);
    EXPECT_EQ(0, L.nrow());
    EXPECT_EQ(0, L.ncol());

    L = Sigma.chol();
    EXPECT_DOUBLE_EQ(Sigma.det(),
                     square(prod(L.diag())))
        << "Sigma = " << endl << Sigma << endl
        << "L = " << endl << L << endl;

    EXPECT_DOUBLE_EQ(Sigma.logdet(),
                     log(Sigma.det()));
  }

  
  TEST_F(SpdMatrixTest, Inv) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    SpdMatrix siginv = Sigma.inv();
    SpdMatrix I(4, 1.0);

    EXPECT_TRUE(MatrixEquals(Sigma * siginv, I))
        << "Sigma = " << endl << Sigma << endl
        << "siginv = " << endl << siginv << endl
        << "Sigma * siginv = " << endl
        << Sigma * siginv << endl;

    SpdMatrix Sigma_copy(Sigma);
    EXPECT_TRUE(MatrixEquals(Sigma, Sigma_copy))
        << "Sigma = " << endl << Sigma << endl
        << "Sigma_copy = " << endl << Sigma_copy;
  }

  TEST_F(SpdMatrixTest, Solve) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    Vector v(4);
    v.randomize();

    Vector x = Sigma.solve(v);
    EXPECT_TRUE(VectorEquals(Sigma * x, v));

    SpdMatrix zero(4);
    bool ok = true;
    x = zero.solve(v, ok);
    EXPECT_FALSE(ok);
    EXPECT_DOUBLE_EQ(x[0], negative_infinity());
    EXPECT_DOUBLE_EQ(x[1], negative_infinity());
    EXPECT_DOUBLE_EQ(x[2], negative_infinity());
    EXPECT_DOUBLE_EQ(x[3], negative_infinity());

    Matrix m(4, 6);
    m.randomize();
    Matrix X = Sigma.solve(m);
    EXPECT_TRUE(MatrixEquals(Sigma * X, m));
  }

  TEST_F(SpdMatrixTest, Reflect) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    EXPECT_NE(Sigma(1, 2), 2.0);
    Sigma(1, 2) = 2.0;
    EXPECT_NE(Sigma(2, 1), 2.0);
    Sigma.reflect();
    EXPECT_DOUBLE_EQ(Sigma(2, 1), 2.0);

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < i; ++j) {
        Sigma(i, j) = 0;
      }
    }
    Sigma.reflect();
    // Check first column
    EXPECT_DOUBLE_EQ(Sigma(0, 1), Sigma(1, 0));
    EXPECT_DOUBLE_EQ(Sigma(0, 2), Sigma(2, 0));
    EXPECT_DOUBLE_EQ(Sigma(0, 3), Sigma(3, 0));

    // Check second column
    EXPECT_DOUBLE_EQ(Sigma(2, 1), Sigma(1, 2));
    EXPECT_DOUBLE_EQ(Sigma(3, 1), Sigma(1, 3));

    // Check third column
    EXPECT_DOUBLE_EQ(Sigma(3, 2), Sigma(2, 3));
  }
  
  TEST_F(SpdMatrixTest, AddOuter) {
    SpdMatrix Sigma(4);
    Sigma.randomize();
    SpdMatrix original_sigma = Sigma;

    Vector v(4);
    v.randomize();

    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(v, 1.7),
        original_sigma + 1.7 * v.outer()));

    cout << "Checking VectorView" << endl;
    Sigma = original_sigma;
    VectorView view(v);
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(view, 1.4),
        original_sigma + 1.4 * v.outer()));

    cout << "Checking ConstVectorView" << endl;
    Sigma = original_sigma;
    const VectorView const_view(v);
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(const_view, 1.9),
        original_sigma + 1.9 * v.outer()));

    cout << "Checking ConstVectorView" << endl;
    Sigma = original_sigma;
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(ConstVectorView(v), 1.9),
        original_sigma + 1.9 * v.outer()));

    cout << "Checking Matrix" << endl;
    Sigma = original_sigma;
    Matrix X(4, 2);
    X.randomize();
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(X, 1.2),
        original_sigma + 1.2 * X * X.transpose()));
    
    Matrix Y(4, 2);
    Y.randomize();
    Sigma = original_sigma;
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer2(X, Y, .81),
        original_sigma + .81 * (X * Y.transpose() + Y * X.transpose())));

    Vector v2(v.size());
    v2.randomize();
    Sigma = original_sigma;
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer2(v, v2, 1.3),
        original_sigma + v.outer(v2, 1.3) + v2.outer(v, 1.3)));
  }

  TEST_F(SpdMatrixTest, AddInner) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    Matrix X(3, 4);
    X.randomize();
    
    SpdMatrix original_sigma = Sigma;
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_inner(X, 1.1),
        original_sigma + X.transpose() * X * 1.1))
        << "Sigma = " << endl << Sigma << endl
        << "Direct = " << endl
        << original_sigma + X.transpose() * X * 1.1
        << endl;

    Vector weights(X.nrow());
    weights.randomize();
    Sigma = original_sigma;
    Matrix W(weights.size(), weights.size(), 0.0);
    W.diag() = weights;
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_inner(X, weights),
        original_sigma + X.transpose() * W * X));

    Matrix Y(3, 4);
    Y.randomize();
    Sigma = original_sigma;
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_inner2(X, Y, .3),
        original_sigma + .3 * (X.transpose() * Y + Y.transpose() * X)));
  }

  TEST_F(SpdMatrixTest, Chol2Inv) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    Chol cholesky(Sigma);
    Matrix L = cholesky.getL();

    EXPECT_TRUE(MatrixEquals(
        chol2inv(L) * Sigma,
        SpdMatrix(4, 1.0)));
  }

  TEST_F(SpdMatrixTest, InvertInplace) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    double ld_sigma = Sigma.logdet();
    SpdMatrix original_sigma = Sigma;
    double ld_siginv = Sigma.invert_inplace();
    EXPECT_NEAR(ld_sigma, -ld_siginv, 1e-7);

    EXPECT_TRUE(MatrixEquals(
        Sigma * original_sigma,
        SpdMatrix(4, 1.0)))
        << "original_sigma = " << endl
        << original_sigma << endl
        << "inverted matrix: " << endl
        << Sigma << endl
        << "product: " << endl
        << original_sigma * Sigma;
  }
  
  TEST_F(SpdMatrixTest, Sandwich) {
    SpdMatrix Sigma(4);
    Sigma.randomize();

    Matrix M(4, 4);
    M.randomize();
    EXPECT_TRUE(MatrixEquals(
        sandwich(M, Sigma),
        M * Sigma * M.transpose()));
  }
  
}  // namespace

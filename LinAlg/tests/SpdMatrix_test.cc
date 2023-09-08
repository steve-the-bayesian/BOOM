#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Selector.hpp"
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
    Selector inc("1001");

    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(v, 1.7),
        original_sigma + 1.7 * v.outer()));

    Sigma = original_sigma;
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(v, inc, 1.7),
        original_sigma + 1.7 * inc.expand(inc.select(v.outer()))));

    Sigma = original_sigma;
    VectorView view(v);
    EXPECT_TRUE(MatrixEquals(
        Sigma.add_outer(view, 1.4),
        original_sigma + 1.4 * v.outer()));

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

    Cholesky cholesky(Sigma);
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

  TEST_F(SpdMatrixTest, KroneckerTest) {
    SpdMatrix A(2);
    SpdMatrix B(3);

    A.randomize();
    B.randomize();
    SpdMatrix K = Kronecker(A, B);

    EXPECT_DOUBLE_EQ(K(0, 0), A(0, 0) * B(0, 0));
    EXPECT_DOUBLE_EQ(K(0, 1), A(0, 0) * B(0, 1));
    EXPECT_DOUBLE_EQ(K(0, 2), A(0, 0) * B(0, 2));
    EXPECT_DOUBLE_EQ(K(1, 0), A(0, 0) * B(1, 0));
    EXPECT_DOUBLE_EQ(K(1, 1), A(0, 0) * B(1, 1));
    EXPECT_DOUBLE_EQ(K(1, 2), A(0, 0) * B(1, 2));
    EXPECT_DOUBLE_EQ(K(2, 0), A(0, 0) * B(2, 0));
    EXPECT_DOUBLE_EQ(K(2, 1), A(0, 0) * B(2, 1));
    EXPECT_DOUBLE_EQ(K(2, 2), A(0, 0) * B(2, 2));

    EXPECT_DOUBLE_EQ(K(0, 3), A(0, 1) * B(0, 0));
    EXPECT_DOUBLE_EQ(K(0, 4), A(0, 1) * B(0, 1));
    EXPECT_DOUBLE_EQ(K(0, 5), A(0, 1) * B(0, 2));
    EXPECT_DOUBLE_EQ(K(1, 3), A(0, 1) * B(1, 0));
    EXPECT_DOUBLE_EQ(K(1, 4), A(0, 1) * B(1, 1));
    EXPECT_DOUBLE_EQ(K(1, 5), A(0, 1) * B(1, 2));
    EXPECT_DOUBLE_EQ(K(2, 3), A(0, 1) * B(2, 0));
    EXPECT_DOUBLE_EQ(K(2, 4), A(0, 1) * B(2, 1));
    EXPECT_DOUBLE_EQ(K(2, 5), A(0, 1) * B(2, 2));

    EXPECT_DOUBLE_EQ(K(3, 0), A(1, 0) * B(0, 0));
    EXPECT_DOUBLE_EQ(K(3, 1), A(1, 0) * B(0, 1));
    EXPECT_DOUBLE_EQ(K(3, 2), A(1, 0) * B(0, 2));
    EXPECT_DOUBLE_EQ(K(4, 0), A(1, 0) * B(1, 0));
    EXPECT_DOUBLE_EQ(K(4, 1), A(1, 0) * B(1, 1));
    EXPECT_DOUBLE_EQ(K(4, 2), A(1, 0) * B(1, 2));
    EXPECT_DOUBLE_EQ(K(5, 0), A(1, 0) * B(2, 0));
    EXPECT_DOUBLE_EQ(K(5, 1), A(1, 0) * B(2, 1));
    EXPECT_DOUBLE_EQ(K(5, 2), A(1, 0) * B(2, 2));

    EXPECT_DOUBLE_EQ(K(3, 3), A(1, 1) * B(0, 0));
    EXPECT_DOUBLE_EQ(K(3, 4), A(1, 1) * B(0, 1));
    EXPECT_DOUBLE_EQ(K(3, 5), A(1, 1) * B(0, 2));
    EXPECT_DOUBLE_EQ(K(4, 3), A(1, 1) * B(1, 0));
    EXPECT_DOUBLE_EQ(K(4, 4), A(1, 1) * B(1, 1));
    EXPECT_DOUBLE_EQ(K(4, 5), A(1, 1) * B(1, 2));
    EXPECT_DOUBLE_EQ(K(5, 3), A(1, 1) * B(2, 0));
    EXPECT_DOUBLE_EQ(K(5, 4), A(1, 1) * B(2, 1));
    EXPECT_DOUBLE_EQ(K(5, 5), A(1, 1) * B(2, 2));
  }

  TEST_F(SpdMatrixTest, TestBlockDiagonal) {
    SpdMatrix m1(2);
    m1.randomize();
    SpdMatrix m2(3);
    m2.randomize();

    SpdMatrix bd = block_diagonal_spd({m1, m2});
    EXPECT_EQ(bd.nrow(), 5);
    EXPECT_DOUBLE_EQ(bd(0, 0), m1(0, 0));
    EXPECT_DOUBLE_EQ(bd(0, 1), m1(0, 1));
    EXPECT_DOUBLE_EQ(bd(1, 0), m1(1, 0));
    EXPECT_DOUBLE_EQ(bd(1, 1), m1(1, 1));

    for (int i = 2; i <= 4; ++i) {
      for (int j = 2; j <= 4; ++j) {
        EXPECT_DOUBLE_EQ(bd(i, j), m2(i-2, j-2));
      }
    }
  }

  TEST_F(SpdMatrixTest, TestScaleOffDiagonal) {
    SpdMatrix X(3);
    X.randomize();
    SpdMatrix Y(X);

    double scale = .3;
    X.scale_off_diagonal(scale);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        if (i == j) {
          EXPECT_DOUBLE_EQ(X(i, j), Y(i, j));
        } else {
          EXPECT_DOUBLE_EQ(X(i, j), Y(i, j) * .3);
        }
      }
    }
  }

  TEST_F(SpdMatrixTest, TestSelfDiagonalAverage) {
    SpdMatrix X(3);
    X.randomize();
    SpdMatrix D(3);
    D.diag() = X.diag();

    SpdMatrix X1 = self_diagonal_average(X, 0);
    SpdMatrix target = X;
    EXPECT_TRUE(MatrixEquals(X1, target))
        << "diagonal_shrinkage = " << 0.0
        << "\n"
        << "X1 = \n" << X1
        << "target = \n" << target;

    SpdMatrix X2 = self_diagonal_average(X, 1.0);
    EXPECT_TRUE(MatrixEquals(X2, D))
        << "diagonal_shrinkage = " << 0.0
        << "\n"
        << "X2 = \n" << X2
        << "target = \n" << D;

    SpdMatrix X3 = self_diagonal_average(X, .7);
    target = .3 * X + .7 * D;
    EXPECT_TRUE(MatrixEquals(X3, target))
        << "diagonal_shrinkage = " << 0.0
        << "\n"
        << "X3 = \n" << X3
        << "target = \n" << target;
  }

}  // namespace

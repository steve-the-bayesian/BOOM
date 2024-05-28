#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/SubMatrix.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MatrixTest : public ::testing::Test {
   protected:
    MatrixTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MatrixTest, Constructors) {
    Matrix empty;
    EXPECT_EQ(0, empty.nrow());
    EXPECT_EQ(0, empty.ncol());

    Matrix single(1, 1);
    EXPECT_EQ(1, single.nrow());
    EXPECT_EQ(1, single.ncol());
    EXPECT_DOUBLE_EQ(0.0, single(0, 0));

    Matrix from_string("1 2 | 5 4");
    EXPECT_EQ(2, from_string.nrow());
    EXPECT_EQ(2, from_string.ncol());
    EXPECT_DOUBLE_EQ(from_string(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(from_string(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(from_string(1, 0), 5.0);
    EXPECT_DOUBLE_EQ(from_string(1, 1), 4.0);
  }

  TEST_F(MatrixTest, FromRowsOrCols) {
    std::vector<Vector> vectors = {
      Vector{1.0, 2.0},
      Vector{3.0, 4.0},
      Vector{5.0, 6.0}
    };

    Matrix m1(vectors, true);
    EXPECT_EQ(m1.nrow(), 3);
    EXPECT_EQ(m1.ncol(), 2);
    EXPECT_DOUBLE_EQ(m1(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(m1(1, 0), 3.0);

    Matrix m2(vectors, false);
    EXPECT_EQ(m2.nrow(), 2);
    EXPECT_EQ(m2.ncol(), 3);
    EXPECT_DOUBLE_EQ(m2(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(m2(1, 0), 2.0);
  }

  TEST_F(MatrixTest, Multiplication) {
    Matrix M(3, 4);
    M.randomize();

    Vector v(4);
    v.randomize();
    Vector product = M * v;
    EXPECT_EQ(3, product.size());
    EXPECT_DOUBLE_EQ(M.row(0).dot(v), product[0]);
    EXPECT_DOUBLE_EQ(M.row(1).dot(v), product[1]);
    EXPECT_DOUBLE_EQ(M.row(2).dot(v), product[2]);

    VectorView view(v);
    product = M * view;
    EXPECT_EQ(3, product.size());
    EXPECT_DOUBLE_EQ(M.row(0).dot(v), product[0]);
    EXPECT_DOUBLE_EQ(M.row(1).dot(v), product[1]);
    EXPECT_DOUBLE_EQ(M.row(2).dot(v), product[2]);

    product = M * ConstVectorView(v);
    EXPECT_EQ(3, product.size());
    EXPECT_DOUBLE_EQ(M.row(0).dot(v), product[0]);
    EXPECT_DOUBLE_EQ(M.row(1).dot(v), product[1]);
    EXPECT_DOUBLE_EQ(M.row(2).dot(v), product[2]);

    Vector v3(3);
    v3.randomize();
    product = M.Tmult(v3);
    EXPECT_EQ(product.size(), 4);
    EXPECT_EQ(product[0], M.col(0).dot(v3));
    EXPECT_EQ(product[1], M.col(1).dot(v3));
    EXPECT_EQ(product[2], M.col(2).dot(v3));
    EXPECT_EQ(product[3], M.col(3).dot(v3));

    VectorView v3_view(v3);
    EXPECT_TRUE(VectorEquals(product, M.Tmult(v3_view)));
    EXPECT_TRUE(VectorEquals(product, M.Tmult(ConstVectorView(v3))));

    Matrix M2(4, 4);
    M2.randomize();

    Matrix MM2 = M * M2;
    EXPECT_EQ(3, MM2.nrow());
    EXPECT_EQ(4, MM2.ncol());
    EXPECT_DOUBLE_EQ(MM2(0, 0), M.row(0).dot(M2.col(0)));
    EXPECT_DOUBLE_EQ(MM2(0, 1), M.row(0).dot(M2.col(1)));
    EXPECT_DOUBLE_EQ(MM2(0, 2), M.row(0).dot(M2.col(2)));
    EXPECT_DOUBLE_EQ(MM2(0, 3), M.row(0).dot(M2.col(3)));
    EXPECT_DOUBLE_EQ(MM2(1, 0), M.row(1).dot(M2.col(0)));
    EXPECT_DOUBLE_EQ(MM2(1, 1), M.row(1).dot(M2.col(1)));
    EXPECT_DOUBLE_EQ(MM2(1, 2), M.row(1).dot(M2.col(2)));
    EXPECT_DOUBLE_EQ(MM2(1, 3), M.row(1).dot(M2.col(3)));
    EXPECT_DOUBLE_EQ(MM2(2, 0), M.row(2).dot(M2.col(0)));
    EXPECT_DOUBLE_EQ(MM2(2, 1), M.row(2).dot(M2.col(1)));
    EXPECT_DOUBLE_EQ(MM2(2, 2), M.row(2).dot(M2.col(2)));
    EXPECT_DOUBLE_EQ(MM2(2, 3), M.row(2).dot(M2.col(3)));

    SpdMatrix V(4);
    V.randomize();
    Matrix MV = M * V;
    EXPECT_EQ(3, MV.nrow());
    EXPECT_EQ(4, MV.ncol());
    EXPECT_DOUBLE_EQ(MV(0, 0), M.row(0).dot(V.col(0)));
    EXPECT_DOUBLE_EQ(MV(0, 1), M.row(0).dot(V.col(1)));
    EXPECT_DOUBLE_EQ(MV(0, 2), M.row(0).dot(V.col(2)));
    EXPECT_DOUBLE_EQ(MV(0, 3), M.row(0).dot(V.col(3)));
    EXPECT_DOUBLE_EQ(MV(1, 0), M.row(1).dot(V.col(0)));
    EXPECT_DOUBLE_EQ(MV(1, 1), M.row(1).dot(V.col(1)));
    EXPECT_DOUBLE_EQ(MV(1, 2), M.row(1).dot(V.col(2)));
    EXPECT_DOUBLE_EQ(MV(1, 3), M.row(1).dot(V.col(3)));
    EXPECT_DOUBLE_EQ(MV(2, 0), M.row(2).dot(V.col(0)));
    EXPECT_DOUBLE_EQ(MV(2, 1), M.row(2).dot(V.col(1)));
    EXPECT_DOUBLE_EQ(MV(2, 2), M.row(2).dot(V.col(2)));
    EXPECT_DOUBLE_EQ(MV(2, 3), M.row(2).dot(V.col(3)));

    EXPECT_TRUE(MatrixEquals(MV, M.multT(V)));

    EXPECT_TRUE(MatrixEquals(M.inner(), M.transpose() * M));
    EXPECT_TRUE(MatrixEquals(M.outer(), M * M.transpose()));
  }

  TEST_F(MatrixTest, Inv) {
    Matrix M(4, 4);
    M.randomize();

    Matrix Minv = M.inv();
    SpdMatrix I(4, 1.0);

    EXPECT_TRUE(MatrixEquals(M * Minv, I))
        << "M = " << endl << M << endl
        << "Minv = " << endl << Minv << endl
        << "M * Minv = " << endl
        << M * Minv << endl;

    Matrix M_copy(M);
    EXPECT_TRUE(MatrixEquals(M, M_copy))
        << "M = " << endl << M << endl
        << "M_copy = " << endl << M_copy;
  }

  TEST_F(MatrixTest, Solve) {
    Matrix M(4, 4);
    M.randomize();

    Vector v(4);
    v.randomize();

    Vector x = M.solve(v);
    EXPECT_TRUE(VectorEquals(M * x, v));

    Matrix M2(4, 6);
    M2.randomize();
    Matrix X = M.solve(M2);
    EXPECT_TRUE(MatrixEquals(M * X, M2));
  }

  template <class MAT1, class MAT2>
  void CheckFieldOperators(const MAT1 &x, const MAT2 &y, const std::string &msg) {
    double a = 1.7;
    Matrix z = x + y;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) + y(0, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) + y(0, 1)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) + y(1, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) + y(1, 1)) << msg;

    z = x - y;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) - y(0, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) - y(0, 1)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) - y(1, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) - y(1, 1)) << msg;

    z = x / y;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) / y(0, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) / y(0, 1)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) / y(1, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) / y(1, 1)) << msg;

    z = x + a;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) + a) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) + a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) + a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) + a) << msg;

    z = x - a;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) - a) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) - a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) - a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) - a) << msg;

    z = a + x;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) + a) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) + a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) + a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) + a) << msg;

    z = a - x;
    EXPECT_DOUBLE_EQ(z(0, 0), a - x(0, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), a - x(0, 1)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), a - x(1, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), a - x(1, 1)) << msg;

    z = x * a;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) * a) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) * a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) * a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) * a) << msg;

    z = x / a;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) / a) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) / a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) / a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) / a) << msg;

    z = a * x;
    EXPECT_DOUBLE_EQ(z(0, 0), x(0, 0) * a) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), x(0, 1) * a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), x(1, 0) * a) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), x(1, 1) * a) << msg;

    z = a / x;
    EXPECT_DOUBLE_EQ(z(0, 0), a / x(0, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(0, 1), a / x(0, 1)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 0), a / x(1, 0)) << msg;
    EXPECT_DOUBLE_EQ(z(1, 1), a / x(1, 1)) << msg;
  }

  TEST_F(MatrixTest, FieldOperators) {
    Matrix x(2, 2);
    Matrix y(2, 2);
    x.randomize();
    y.randomize();
    SubMatrix xview(x);
    SubMatrix yview(y);
    ConstSubMatrix cxview(x);
    ConstSubMatrix cyview(y);

    CheckFieldOperators(x, y, "mat, mat");
    CheckFieldOperators(x, yview, "mat, view");
    CheckFieldOperators(x, cyview, "mat, const view");
    CheckFieldOperators(xview, y, "view, mat");
    CheckFieldOperators(xview, yview, "view, view");
    CheckFieldOperators(xview, cyview, "view, const view");
    CheckFieldOperators(cxview, y, "const view, mat");
    CheckFieldOperators(cxview, yview, "const view, view");
    CheckFieldOperators(cxview, cyview, "const view, const view");
  }

  TEST_F(MatrixTest, Trace) {
    Matrix M(4, 4);
    M.randomize();
    EXPECT_DOUBLE_EQ(M.trace(), trace(M));
    EXPECT_DOUBLE_EQ(M.trace(), sum(M.diag()));
  }

  TEST_F(MatrixTest, AddOuter) {
    Matrix M(4, 4);
    M.randomize();
    Matrix original_matrix = M;

    Vector v(4);
    v.randomize();

    EXPECT_TRUE(MatrixEquals(
        M.add_outer(v, v, 1.7),
        original_matrix + 1.7 * v.outer()));

    cout << "Checking VectorView" << endl;
    M = original_matrix;
    VectorView view(v);
    EXPECT_TRUE(MatrixEquals(
        M.add_outer(view, view, 1.4),
        original_matrix + 1.4 * v.outer()));

    cout << "Checking ConstVectorView" << endl;
    M = original_matrix;
    const VectorView const_view(v);
    EXPECT_TRUE(MatrixEquals(
        M.add_outer(const_view, const_view, 1.9),
        original_matrix + 1.9 * v.outer()));

    cout << "Checking ConstVectorView" << endl;
    M = original_matrix;
    EXPECT_TRUE(MatrixEquals(
        M.add_outer(const_view, const_view, 1.9),
        original_matrix + 1.9 * v.outer()));

  }

  // TEST_F(SpdMatrixTest, AddInner) {
  //   SpdMatrix Sigma(4);
  //   Sigma.randomize();

  //   Matrix X(3, 4);
  //   X.randomize();

  //   SpdMatrix original_sigma = Sigma;
  //   EXPECT_TRUE(MatrixEquals(
  //       Sigma.add_inner(X, 1.1),
  //       original_sigma + X.transpose() * X * 1.1))
  //       << "Sigma = " << endl << Sigma << endl
  //       << "Direct = " << endl
  //       << original_sigma + X.transpose() * X * 1.1
  //       << endl;

  //   Vector weights(X.nrow());
  //   weights.randomize();
  //   Sigma = original_sigma;
  //   Matrix W(weights.size(), weights.size(), 0.0);
  //   W.diag() = weights;
  //   EXPECT_TRUE(MatrixEquals(
  //       Sigma.add_inner(X, weights),
  //       original_sigma + X.transpose() * W * X));

  //   Matrix Y(3, 4);
  //   Y.randomize();
  //   Sigma = original_sigma;
  //   EXPECT_TRUE(MatrixEquals(
  //       Sigma.add_inner2(X, Y, .3),
  //       original_sigma + .3 * (X.transpose() * Y + Y.transpose() * X)));

  // }

  TEST_F(MatrixTest, Operators) {
    Matrix M(3, 3);
    M.randomize();
    Matrix original_M = M;

    M *= 2.0;
    EXPECT_DOUBLE_EQ(M(1, 2), original_M(1, 2) * 2.0);

    M = original_M;
    ////////////////////////////////////
  }

  TEST_F(MatrixTest, Norms) {
    Matrix M(2, 2);
    M.randomize();

    EXPECT_DOUBLE_EQ(M.abs_norm(),
                     fabs(M(0, 0)) + fabs(M(0, 1)) + fabs(M(1, 0)) + fabs(M(1, 1)));

    EXPECT_DOUBLE_EQ(M.sumsq(), sum(el_mult(M, M)));
  }

  TEST_F(MatrixTest, LowerTriangular) {

    Matrix L(3, 3);
    L.randomize();
    L(0, 1) = L(0, 2) = L(1, 2) = 0.0;

    Vector v(3);
    EXPECT_TRUE(VectorEquals(Lmult(L, v), L * v));
    EXPECT_TRUE(VectorEquals(LTmult(L, v), L.transpose() * v));

    EXPECT_TRUE(VectorEquals(Lsolve(L, v), L.inv() * v));
    Vector original_v = v;
    EXPECT_TRUE(VectorEquals(LTsolve_inplace(L, v),
                             L.transpose().inv() * original_v));
    v = original_v;
    EXPECT_TRUE(VectorEquals(Lsolve_inplace(L, v), L.inv() * original_v));
    v = original_v;

    Matrix B(3, 3);
    B.randomize();
    EXPECT_TRUE(MatrixEquals(Lsolve(L, B), L.inv() * B));
    Matrix original_B = B;
    EXPECT_TRUE(MatrixEquals(Lsolve_inplace(L, B), L.inv() * original_B));
    B = original_B;
    EXPECT_TRUE(MatrixEquals(LTsolve_inplace(L, B),
                             L.transpose().inv() * original_B));
    B = original_B;
    EXPECT_TRUE(MatrixEquals(Linv(L), L.inv()));

    Matrix U = L.transpose();
    EXPECT_TRUE(VectorEquals(Umult(U, v), U * v));
    EXPECT_TRUE(MatrixEquals(Umult(U, B), U * B));
    EXPECT_TRUE(VectorEquals(Usolve(U, v), U.inv() * v));
    EXPECT_TRUE(VectorEquals(Usolve_inplace(U, v), U.inv() * original_v));
    v = original_v;

    EXPECT_TRUE(MatrixEquals(Usolve(U, B), U.inv() * B));
    EXPECT_TRUE(MatrixEquals(Usolve_inplace(U, B), U.inv() * original_B));
    B = original_B;
    EXPECT_TRUE(MatrixEquals(Uinv(U), U.inv()));
  }

  TEST_F(MatrixTest, Determinant) {
    Matrix A(2, 2);
    A.randomize();
    double a = A(0, 0);
    double b = A(0, 1);
    double c = A(1, 0);
    double d = A(1, 1);
    double epsilon = 1e-8;
    EXPECT_NEAR(A.det(), a * d - b * c, epsilon);
    EXPECT_NEAR(A.logdet(), log(fabs(a*d - b*c)), epsilon);
  }

  TEST_F(MatrixTest, ConditionNumberTest) {
    Matrix A(2, 2);
    A = 1.0;
    EXPECT_FALSE(std::isfinite(A.condition_number()));

    for (int i = 0; i < 1000; ++i) {
      A.randomize();
      EXPECT_GT(A.condition_number(), 0.0);
      EXPECT_LT(A.condition_number(), 10000.0);
    }
  }

  TEST_F(MatrixTest, BlockDiagonal) {
    Matrix A(2, 2);
    A.randomize();

    Matrix B(2, 3);
    B.randomize();

    Matrix M = block_diagonal(A, B);
    EXPECT_EQ(4, M.nrow());
    EXPECT_EQ(5, M.ncol());
    EXPECT_DOUBLE_EQ(M(0, 0), A(0, 0));
    EXPECT_DOUBLE_EQ(M(0, 1), A(0, 1));
    EXPECT_DOUBLE_EQ(M(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(M(0, 3), 0.0);
    EXPECT_DOUBLE_EQ(M(0, 4), 0.0);

    EXPECT_DOUBLE_EQ(M(1, 0), A(1, 0));
    EXPECT_DOUBLE_EQ(M(1, 1), A(1, 1));
    EXPECT_DOUBLE_EQ(M(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(M(1, 3), 0.0);
    EXPECT_DOUBLE_EQ(M(1, 4), 0.0);

    EXPECT_DOUBLE_EQ(M(2, 0), 0.0);
    EXPECT_DOUBLE_EQ(M(2, 1), 0.0);
    EXPECT_DOUBLE_EQ(M(2, 2), B(0, 0));
    EXPECT_DOUBLE_EQ(M(2, 3), B(0, 1));
    EXPECT_DOUBLE_EQ(M(2, 4), B(0, 2));

    EXPECT_DOUBLE_EQ(M(3, 0), 0.0);
    EXPECT_DOUBLE_EQ(M(3, 1), 0.0);
    EXPECT_DOUBLE_EQ(M(3, 2), B(1, 0));
    EXPECT_DOUBLE_EQ(M(3, 3), B(1, 1));
    EXPECT_DOUBLE_EQ(M(3, 4), B(1, 2));
  }

  TEST_F(MatrixTest, InnerProduct) {
    Matrix square(3, 3);
    square.randomize();
    Vector weights(3);
    weights.randomize();
    EXPECT_TRUE(MatrixEquals(
        square.inner(weights),
        square.transpose() * DiagonalMatrix(weights) * square));

    Matrix skinny(8, 3);
    Vector skinny_weights(8);
    skinny.randomize();
    skinny_weights.randomize();
    EXPECT_TRUE(MatrixEquals(
        skinny.inner(skinny_weights),
        skinny.transpose() * DiagonalMatrix(skinny_weights) * skinny));

    Matrix fat(3, 8);
    Vector fat_weights(fat.nrow());
    fat.randomize();
    fat_weights.randomize();
    EXPECT_TRUE(MatrixEquals(
        fat.inner(fat_weights),
        fat.transpose() * DiagonalMatrix(fat_weights) * fat));
  }

  TEST_F(MatrixTest, KroneckerProduct) {
    Matrix A(2, 3);
    A.randomize();
    Matrix B(5, 4);
    B.randomize();
    Matrix K = Kronecker(A, B);
    EXPECT_EQ(K.nrow(), 2 * 5);
    EXPECT_EQ(K.ncol(), 3 * 4);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        EXPECT_TRUE(MatrixEquals(const_block(K, i, j, 5, 4),
                                 A(i, j) * B));
      }
    }
  }

  TEST_F(MatrixTest, Vectorize) {
    Matrix A = rbind(Vector{1, 2, 3}, Vector{4, 5, 6});
    EXPECT_TRUE(VectorEquals(vec(A), Vector{1, 4, 2, 5, 3, 6}));
  }


  TEST_F(MatrixTest, RowAndColSums) {
    Matrix A(3, 4);
    A.randomize();

    Vector row_sums = A.row_sums();
    EXPECT_DOUBLE_EQ(row_sums[0], A(0, 0) + A(0, 1) + A(0, 2) + A(0, 3));
    EXPECT_DOUBLE_EQ(row_sums[1], A(1, 0) + A(1, 1) + A(1, 2) + A(1, 3));
    EXPECT_DOUBLE_EQ(row_sums[2], A(2, 0) + A(2, 1) + A(2, 2) + A(2, 3));

    Vector col_sums = A.col_sums();
    EXPECT_DOUBLE_EQ(col_sums[0], A(0, 0) + A(1, 0) + A(2, 0));
  }

  TEST_F(MatrixTest, RelativeDistance) {
    Matrix A(3, 4);
    A.randomize();

    Matrix B = A;
    double d = relative_distance(A, B);
    EXPECT_NEAR(d, 0.0, 1e-8);

    B(1, 2) += .10;
    int i, j;
    d = relative_distance(A, B, i, j);
    EXPECT_EQ(i, 1);
    EXPECT_EQ(j, 2);
    EXPECT_NEAR(d,
                .5 * .1 / (A(1, 2) + B(1,2)),
                1e-8);

  }

  TEST_F(MatrixTest, Stacking) {
    Matrix A(2, 3);
    A.row(0) = Vector{1, 2, 3};
    A.row(1) = Vector{4, 5, 6};

    EXPECT_TRUE(VectorEquals(stack_rows(A), Vector{1, 2, 3, 4, 5, 6}));
    EXPECT_TRUE(VectorEquals(stack_columns(A), Vector{1, 4, 2, 5, 3, 6}));
  }

}  // namespace

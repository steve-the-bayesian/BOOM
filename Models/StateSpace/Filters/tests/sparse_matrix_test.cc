#include "gtest/gtest.h"

#include "Models/StateSpace/Filters/SparseMatrix.hpp"

#include "test_utils/test_utils.hpp"

namespace {

  using namespace BOOM;
  using std::endl;

  class SparseMatrixTest : public ::testing::Test {
   protected:
    SparseMatrixTest() {
      GlobalRng::rng.seed(8675309);
    }
  };
  
  void CheckSparseMatrixBlock(
      const Ptr<SparseMatrixBlock> &sparse,
      const Matrix &dense) {
    EXPECT_TRUE(MatrixEquals(sparse->dense(), dense))
        << "sparse->dense() = " << endl
        << sparse->dense() << endl
        << "dense = " << endl
        << dense << endl;
    
    EXPECT_EQ(sparse->nrow(), dense.nrow())
        << endl << sparse->dense() << endl << dense;
    EXPECT_EQ(sparse->ncol(), dense.ncol())
        << endl << sparse->dense() << endl << dense;

    Vector rhs_vector(dense.ncol());
    rhs_vector.randomize();
    Vector lhs_vector(dense.nrow());
    sparse->multiply(VectorView(lhs_vector), rhs_vector);
    EXPECT_TRUE(VectorEquals(lhs_vector, dense * rhs_vector))
        << endl << sparse->dense() << endl << dense << endl
        << "rhs = " << rhs_vector << endl
        << "sparse * rhs = " << lhs_vector << endl
        << "dense * rhs = " << dense * rhs_vector << endl;

    lhs_vector.randomize();
    Vector original_lhs = lhs_vector;
    rhs_vector.randomize();
    sparse->multiply_and_add(VectorView(lhs_vector), rhs_vector);
    EXPECT_TRUE(VectorEquals(lhs_vector, original_lhs + dense * rhs_vector))
        << endl << sparse->dense() << endl << dense << endl
        << "rhs = " << rhs_vector << endl
        << "lhs = " << original_lhs << endl
        << "lhs + sparse * rhs = " << lhs_vector << endl
        << "lhs + dense * rhs = " << lhs_vector + dense * rhs_vector << endl;

    Vector rhs_tmult_vector(dense.nrow());
    Vector lhs_tmult_vector(dense.ncol());
    sparse->Tmult(VectorView(lhs_tmult_vector), rhs_tmult_vector);
    EXPECT_TRUE(VectorEquals(lhs_tmult_vector,
                             rhs_tmult_vector * dense));
    
    // Only check multiply_inplace and friends if the matrix is square.
    if (dense.nrow() == dense.ncol()) {
      Vector original_rhs = rhs_vector;
      sparse->multiply_inplace(VectorView(rhs_vector));
      EXPECT_TRUE(VectorEquals(rhs_vector, dense * original_rhs))
        << endl << sparse->dense() << endl << dense << endl
        << "rhs = " << original_rhs << endl
        << "sparse->multiply_inplace(rhs) = " << rhs_vector << endl
        << "dense * rhs = " << dense * original_rhs << endl;

      Matrix rhs_matrix(dense.ncol(), dense.ncol());
      rhs_matrix.randomize();
      Matrix original_rhs_matrix = rhs_matrix;
      sparse->matrix_multiply_inplace(SubMatrix(rhs_matrix));
      EXPECT_TRUE(MatrixEquals(rhs_matrix, dense * original_rhs_matrix))
        << endl << sparse->dense() << endl << dense << endl
        << "rhs = " << original_rhs_matrix << endl
        << "sparse->matrix_multiply_inplace(rhs) = " << rhs_matrix << endl
        << "dense * rhs = " << dense * original_rhs_matrix << endl;

      rhs_matrix.randomize();
      original_rhs_matrix = rhs_matrix;
      sparse->matrix_transpose_premultiply_inplace(SubMatrix(rhs_matrix));
      EXPECT_TRUE(MatrixEquals(rhs_matrix,
                               original_rhs_matrix * dense.transpose()))
        << endl << sparse->dense() << endl << dense << endl
        << "rhs = " << endl << original_rhs_matrix << endl
        << "sparse->matrix_transpose_multiply_inplace(rhs) = "
        << endl << rhs_matrix << endl
        << "rhs * dense.transpose() = " << endl
        << original_rhs_matrix * dense.transpose() << endl;
    }

    Matrix summand(dense.nrow(), dense.ncol());
    summand.randomize();
    Matrix original_summand(summand);
    sparse->add_to(SubMatrix(summand));
    EXPECT_TRUE(MatrixEquals(summand, dense + original_summand))
        << endl << sparse->dense() << endl << dense << endl
        << "B = " << original_summand << endl
        << "sparse->add_to(B) = " << summand << endl
        << "dense + B = " << dense + original_summand << endl;

    SpdMatrix inner = sparse->inner();
    EXPECT_TRUE(MatrixEquals(inner, dense.inner()))
        << "inner = " << endl << inner << endl
        << "dense.inner() = " << endl
        << dense.inner();
  }

  void CheckLeftInverse(const Ptr<SparseMatrixBlock> &block,
                        const Vector &rhs) {
    BlockDiagonalMatrix mat;
    mat.add_block(block);
    
    Vector lhs = mat.left_inverse(rhs);
    Vector rhs_new = mat * lhs;

    EXPECT_TRUE(VectorEquals(rhs, rhs_new))
        << "Vectors were not equal." << endl
        << rhs << endl
        << rhs_new;
  }
  
  TEST_F(SparseMatrixTest, LeftInverseIdentity) {
    NEW(IdentityMatrix, mat)(3);
    Vector x(3);
    x.randomize();
    CheckLeftInverse(mat, x);
  }

  TEST_F(SparseMatrixTest, LeftInverseSkinnyColumn) {
    NEW(FirstElementSingleColumnMatrix, column)(12);
    Vector errors(1);
    errors.randomize();
    Vector x(12);
    x[0] = errors[0];
    CheckLeftInverse(column, x);
  }

  TEST_F(SparseMatrixTest, IdentityMatrix) {
    NEW(IdentityMatrix, I3)(3);
    SpdMatrix I3_dense(3, 1.0);
    CheckSparseMatrixBlock(I3, I3_dense);

    NEW(IdentityMatrix, I1)(1);
    SpdMatrix I1_dense(1, 1.0);
    CheckSparseMatrixBlock(I1, I1_dense);
  }

  TEST_F(SparseMatrixTest, LocalTrend) {
    NEW(LocalLinearTrendMatrix, T)();
    Matrix Tdense = T->dense();
    EXPECT_TRUE(VectorEquals(Tdense.row(0), Vector{1, 1}));
    EXPECT_TRUE(VectorEquals(Tdense.row(1), Vector{0, 1}));
    CheckSparseMatrixBlock(T, Tdense);
  }

  TEST_F(SparseMatrixTest, DenseMatrixTest) {
    Matrix square(4, 4);
    square.randomize();
    NEW(DenseMatrix, square_kalman)(square);
    CheckSparseMatrixBlock(square_kalman, square);

    Matrix rectangle(3, 4);
    rectangle.randomize();
    NEW(DenseMatrix, rectangle_kalman)(rectangle);
    CheckSparseMatrixBlock(rectangle_kalman, rectangle);
  }

  TEST_F(SparseMatrixTest, SpdTest) {
    SpdMatrix spd(3);
    spd.randomize();

    NEW(DenseSpd, spd_kalman)(spd);
    CheckSparseMatrixBlock(spd_kalman, spd);

    NEW(SpdParams, sparams)(spd);
    NEW(DenseSpdParamView, spd_view)(sparams);
    CheckSparseMatrixBlock(spd_view, spd);
  }

  TEST_F(SparseMatrixTest, Diagonal) {
    Vector values(4);
    values.randomize();
    
    NEW(DiagonalMatrixBlock, diag)(values);
    Matrix D(4, 4, 0.0);
    D.set_diag(values);

    CheckSparseMatrixBlock(diag, D);

    NEW(VectorParams, vprm)(values);
    NEW(DiagonalMatrixBlockVectorParamView, diag_view)(vprm);
    CheckSparseMatrixBlock(diag_view, D);
  }

  TEST_F(SparseMatrixTest, Seasonal) {
    NEW(SeasonalStateSpaceMatrix, seasonal)(4);
    Matrix seasonal_dense(3, 3, 0.0);
    seasonal_dense.row(0) = -1;
    seasonal_dense.subdiag(1) = 1.0;

    CheckSparseMatrixBlock(seasonal, seasonal_dense);
  }

  TEST_F(SparseMatrixTest, AutoRegression) {
    Vector elements(4);
    elements.randomize();
    NEW(GlmCoefs, rho)(elements);
    NEW(AutoRegressionTransitionMatrix, rho_kalman)(rho);
    Matrix rho_dense(4, 4);
    rho_dense.row(0) = elements;
    rho_dense.subdiag(1) = 1.0;

    CheckSparseMatrixBlock(rho_kalman, rho_dense);
  }

  TEST_F(SparseMatrixTest, EmptyTest) {
    Matrix empty;
    NEW(EmptyMatrix, empty_kalman)();
    CheckSparseMatrixBlock(empty_kalman, empty);
  }

  TEST_F(SparseMatrixTest, ConstantTest) {
    SpdMatrix dense(4, 8.7);
    NEW(ConstantMatrix, sparse)(4, 8.7);
    CheckSparseMatrixBlock(sparse, dense);

    NEW(UnivParams, prm)(8.7);
    NEW(ConstantMatrixParamView, sparse_view)(4, prm);
    CheckSparseMatrixBlock(sparse_view, dense);
  }

  TEST_F(SparseMatrixTest, ZeroTest) {
    NEW(ZeroMatrix, sparse)(7);
    Matrix dense(7, 7, 0.0);
    CheckSparseMatrixBlock(sparse, dense);
  }

  TEST_F(SparseMatrixTest, ULC) {
    NEW(UpperLeftCornerMatrix, sparse)(5, 19.2);
    Matrix dense(5, 5, 0.0);
    dense(0, 0) = 19.2;
    CheckSparseMatrixBlock(sparse, dense);

    NEW(UnivParams, prm)(19.2);
    NEW(UpperLeftCornerMatrixParamView, sparse_view)(5, prm);
    CheckSparseMatrixBlock(sparse_view, dense);
  }

  TEST_F(SparseMatrixTest, FirstElementSingleColumnMatrixTest) {
    NEW(FirstElementSingleColumnMatrix, sparse)(7);
    Matrix dense(7, 1, 0.0);
    dense(0, 0) = 1.0;
    CheckSparseMatrixBlock(sparse, dense);
  }

  TEST_F(SparseMatrixTest, ZeroPaddedIdTest) {
    NEW(ZeroPaddedIdentityMatrix, sparse)(20, 4);
    Matrix dense(20, 4);
    dense.set_diag(1.0);
    CheckSparseMatrixBlock(sparse, dense);
  }

  TEST_F(SparseMatrixTest, SingleSparseDiagonalElementMatrixTest) {
    NEW(SingleSparseDiagonalElementMatrix, sparse)(12, 18.7, 5);
    Matrix dense(12, 12, 0.0);
    dense(5, 5) = 18.7;
    CheckSparseMatrixBlock(sparse, dense);

    NEW(UnivParams, prm)(18.7);
    NEW(SingleSparseDiagonalElementMatrixParamView, sparse_view)(12, prm, 5);
    CheckSparseMatrixBlock(sparse_view, dense);
  }

  TEST_F(SparseMatrixTest, SingleElementInFirstRowTest) {
    NEW(SingleElementInFirstRow, sparse_square)(5, 5, 3, 12.9);
    Matrix dense(5, 5, 0.0);
    dense(0, 3) = 12.9;
    CheckSparseMatrixBlock(sparse_square, dense);

    NEW(SingleElementInFirstRow, sparse_rectangle)(5, 8, 0, 99.99);
    Matrix wide(5, 8, 0.0);
    wide(0, 0) = 99.99;
    CheckSparseMatrixBlock(sparse_rectangle, wide);

    NEW(SingleElementInFirstRow, sparse_tall)(20, 4, 2, 13.7);
    Matrix tall(20, 4, 0.0);
    tall(0, 2) = 13.7;
    CheckSparseMatrixBlock(sparse_tall, tall);
  }

  TEST_F(SparseMatrixTest, UpperLeftDiagonalTest) {
    std::vector<Ptr<UnivParams>> params;
    params.push_back(new UnivParams(3.2));
    params.push_back(new UnivParams(1.7));
    params.push_back(new UnivParams(-19.8));

    NEW(UpperLeftDiagonalMatrix, sparse)(params, 17);
    Matrix dense(17, 17, 0.0);
    for (int i = 0; i < params.size(); ++i) {
      dense(i, i) = params[i]->value();
    }
    CheckSparseMatrixBlock(sparse, dense);

    Vector scale_factor(17);
    scale_factor.randomize();
    dense.diag() *= scale_factor;
    NEW(UpperLeftDiagonalMatrix, sparse2)(
        params, 17, ConstVectorView(scale_factor, 0, 3));
    CheckSparseMatrixBlock(sparse2, dense);
  }

  TEST_F(SparseMatrixTest, IdenticalRowsMatrixTest) {
    SparseVector row(20);
    row[0] = 8;
    row[17] = 6;
    row[12] = 7;
    row[9] = 5;
    row[3] = 3;
    row[1] = 0;
    row[2] = 9;
    NEW(IdenticalRowsMatrix, sparse)(row, 20);
    Matrix dense(20, 20, 0.0);
    dense.col(0) = 8;
    dense.col(17) = 6;
    dense.col(12) = 7;
    dense.col(9) = 5;
    dense.col(3) = 3;
    dense.col(1) = 0;
    dense.col(2) = 9;
    CheckSparseMatrixBlock(sparse, dense);
  }

  Matrix ConstraintMatrix(int dim) {
    Matrix ans(dim, dim, -1.0 / dim);
    ans.diag() += 1.0;
    return ans;
  }

  TEST_F(SparseMatrixTest, ConstraintMatrixTest) {
    Matrix dense = ConstraintMatrix(7);
    NEW(EffectConstraintMatrix, sparse)(7);
    CheckSparseMatrixBlock(sparse, dense);
  }
  
  TEST_F(SparseMatrixTest, EffectConstrainedMatrixBlockTest) {
    NEW(SeasonalStateSpaceMatrix, seasonal)(4);
    NEW(EffectConstrainedMatrixBlock, constrained_seasonal)(
        seasonal);

    Matrix constraint_matrix(3, 3, -1.0/3);
    constraint_matrix.diag() = (2.0 / 3);
    
    EXPECT_TRUE(MatrixEquals(constraint_matrix, ConstraintMatrix(3)));
    
    CheckSparseMatrixBlock(
        constrained_seasonal, seasonal->dense() * ConstraintMatrix(3));
  }

  TEST_F(SparseMatrixTest, GenericSparseMatrixBlockTest) {
    NEW(GenericSparseMatrixBlock, sparse)(12, 18);
    (*sparse)(3, 7) = 19;
    (*sparse)(5, 2) = -4;

    Matrix dense(12, 18, 0.0);
    dense(3, 7) = 19;
    dense(5, 2) = -4;
    CheckSparseMatrixBlock(sparse, dense);


    NEW(GenericSparseMatrixBlock, sparse_square)(7, 7);
    (*sparse_square)(2, 5) = 17.4;
    (*sparse_square)(4, 0) = -83;
    Matrix square_dense(7, 7, 0.0);
    square_dense(2, 5) = 17.4;
    square_dense(4, 0) = -83;
    CheckSparseMatrixBlock(sparse_square, square_dense);
  }

  void CheckSparseKalmanMatrix(const SparseKalmanMatrix &sparse) {
    Matrix dense = sparse.dense();
    EXPECT_EQ(dense.nrow(), sparse.nrow());
    EXPECT_EQ(dense.ncol(), sparse.ncol());

    Vector v(sparse.ncol());
    v.randomize();
    EXPECT_TRUE(VectorEquals(sparse * v, dense * v));
    EXPECT_TRUE(VectorEquals(sparse * VectorView(v), dense * VectorView(v)));
    EXPECT_TRUE(VectorEquals(sparse * ConstVectorView(v),
                             dense * ConstVectorView(v)));
    
    Vector tv(sparse.nrow());
    EXPECT_TRUE(VectorEquals(sparse.Tmult(tv), dense.Tmult(tv)));
    EXPECT_TRUE(VectorEquals(sparse.Tmult(VectorView(tv)),
                             dense.Tmult(VectorView(tv))));
    EXPECT_TRUE(VectorEquals(sparse.Tmult(ConstVectorView(tv)),
                             dense.Tmult(ConstVectorView(tv))));

    SpdMatrix V(sparse.ncol());
    V.randomize();
    SpdMatrix originalV = V;
    if (sparse.nrow() == sparse.ncol()) {
      sparse.sandwich_inplace(V);
      EXPECT_TRUE(MatrixEquals(V, dense * originalV * dense.transpose()));

      SpdMatrix tV(sparse.nrow());
      tV.randomize();
      SpdMatrix original_tV = tV;
      sparse.sandwich_inplace_transpose(tV);
      EXPECT_TRUE(MatrixEquals(tV, dense.transpose() * original_tV * dense));
    }

    EXPECT_TRUE(MatrixEquals(sparse.sandwich(V), dense * V * dense.transpose()));

    Matrix summand(sparse.nrow(), sparse.ncol());
    summand.randomize();
    Matrix original_summand = summand;
    EXPECT_TRUE(MatrixEquals(sparse.add_to(summand), original_summand + dense));


    EXPECT_TRUE(MatrixEquals(dense.inner(), sparse.inner()))
        << "Dense inner product: " << endl
        << dense.inner() << endl
        << "Sparse inner product: " << endl
        << sparse.inner() << endl;
  }
  
  TEST_F(SparseMatrixTest, BlockDiagonalMatrixTest) {
    BlockDiagonalMatrix sparse;
    sparse.add_block(new LocalLinearTrendMatrix);
    sparse.add_block(new SeasonalStateSpaceMatrix(4));
    CheckSparseKalmanMatrix(sparse);
  }

  TEST_F(SparseMatrixTest, SparseVerticalStripMatrixTest) {
    SparseVerticalStripMatrix sparse;
    int nrows = 8;
    SparseVector trend(2);
    EXPECT_EQ(2, trend.size());
    trend[0] = 1.0;
    sparse.add_block(new IdenticalRowsMatrix(trend, nrows));
    EXPECT_EQ(2, sparse.ncol());
    
    SparseVector seasonal(4);
    seasonal[0] = 1.0;
    sparse.add_block(new IdenticalRowsMatrix(seasonal, nrows));
    EXPECT_EQ(6, sparse.ncol());
    EXPECT_EQ(8, sparse.nrow());

    Matrix dense = sparse.dense();
    Vector zero(nrows, 0.0);
    Vector one(nrows, 1.0);
    EXPECT_EQ(dense.nrow(), nrows);
    EXPECT_EQ(dense.ncol(), 6);
    EXPECT_TRUE(VectorEquals(dense.col(0), one));
    EXPECT_TRUE(VectorEquals(dense.col(1), zero));
    EXPECT_TRUE(VectorEquals(dense.col(2), one));
    EXPECT_TRUE(VectorEquals(dense.col(3), zero));
    EXPECT_TRUE(VectorEquals(dense.col(4), zero));
    EXPECT_TRUE(VectorEquals(dense.col(5), zero));
    CheckSparseKalmanMatrix(sparse);
  }
  
}  // namespace

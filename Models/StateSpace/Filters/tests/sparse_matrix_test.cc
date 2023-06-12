#include "gtest/gtest.h"

#include "Models/StateSpace/Filters/SparseMatrix.hpp"
#include "Models/StateSpace/StateModels/SemilocalLinearTrend.hpp"
#include "Models/StateSpace/StateModels/SeasonalStateModel.hpp"
#include "Models/StateSpace/AggregatedStateSpaceRegression.hpp"
#include "Models/TimeSeries/ArmaModel.hpp"

#include "distributions.hpp"
#include "LinAlg/DiagonalMatrix.hpp"

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
    sparse->add_to_block(SubMatrix(summand));
    EXPECT_TRUE(MatrixEquals(summand, dense + original_summand))
        << endl << sparse->dense() << endl << dense << endl
        << "B = " << original_summand << endl
        << "sparse->add_to_block(B) = " << summand << endl
        << "dense + B = " << dense + original_summand << endl;

    SpdMatrix inner = sparse->inner();
    EXPECT_TRUE(MatrixEquals(inner, dense.inner()))
        << "inner = " << endl << inner << endl
        << "dense.inner() = " << endl
        << dense.inner();

    Vector weights(sparse->nrow());
    weights.randomize();
    EXPECT_TRUE(MatrixEquals(sparse->inner(weights), dense.inner(weights)))
        << "dense inner product: " << endl
        << dense.inner(weights) << endl
        << "sparse inner product: " << endl
        << sparse->inner(weights);
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

    Vector weights(dense.nrow());
    weights.randomize();
    EXPECT_TRUE(MatrixEquals(
        dense.Tmult(DiagonalMatrix(weights) * dense),
        sparse.inner(weights)));
  }

  TEST_F(SparseMatrixTest, LeftInverseIdentity) {
    NEW(IdentityMatrix, mat)(3);
    Vector x(3);
    x.randomize();
  }

  TEST_F(SparseMatrixTest, LeftInverseSkinnyColumn) {
    NEW(FirstElementSingleColumnMatrix, column)(12);
    Vector errors(1);
    errors.randomize();
    Vector x(12);
    x[0] = errors[0];
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

  TEST_F(SparseMatrixTest, SemilocalLinearTrendMatrixTest) {
    NEW(UnivParams, phi)(.7);
    NEW(SemilocalLinearTrendMatrix, T)(phi);
    Matrix dense = T->dense();
    EXPECT_TRUE(VectorEquals(dense.row(0), Vector{1, 1, 0}));
    EXPECT_TRUE(VectorEquals(dense.row(1), Vector{0, .7, .3}));
    EXPECT_TRUE(VectorEquals(dense.row(2), Vector{0, 0, 1}));
    CheckSparseMatrixBlock(T, dense);
  }

  TEST_F(SparseMatrixTest, DenseMatrixTest) {
    Matrix square(4, 4);
    square.randomize();
    NEW(DenseMatrix, square_kalman)(square);
    CheckSparseMatrixBlock(square_kalman, square);

    Matrix fat_rectangle(3, 8);
    fat_rectangle.randomize();
    NEW(DenseMatrix, fat_rectangle_kalman)(fat_rectangle);
    CheckSparseMatrixBlock(fat_rectangle_kalman, fat_rectangle);

    Matrix skinny_rectangle(8, 3);
    skinny_rectangle.randomize();
    NEW(DenseMatrix, skinny_rectangle_kalman)(skinny_rectangle);
    CheckSparseMatrixBlock(skinny_rectangle_kalman, skinny_rectangle);

    // Check that inner_product works when there are multiple blocks.
    BlockDiagonalMatrix skinny_blocks;
    skinny_blocks.add_block(skinny_rectangle_kalman);
    Matrix second_skinny_rectangle(5, 2);
    second_skinny_rectangle.randomize();
    NEW(DenseMatrix, second_skinny_rectangle_kalman)(second_skinny_rectangle);
    skinny_blocks.add_block(second_skinny_rectangle_kalman);
    EXPECT_EQ(skinny_blocks.nrow(),
              skinny_rectangle_kalman->nrow()
              + second_skinny_rectangle_kalman->nrow());
    Vector skinny_blocks_weights(skinny_blocks.nrow());
    skinny_blocks_weights.randomize();
    EXPECT_TRUE(MatrixEquals(
        skinny_blocks.inner(skinny_blocks_weights),
        skinny_blocks.dense().inner(skinny_blocks_weights)));
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

  TEST_F(SparseMatrixTest, ArmaTransition) {
    Vector coefficients = {.8, .3, 0, 0};
    NEW(ArmaStateSpaceTransitionMatrix, T)(coefficients);
    Matrix dense = T->dense();
    EXPECT_TRUE(VectorEquals(dense.col(0), coefficients));
    EXPECT_TRUE(VectorEquals(dense.col(1), Vector{1, 0, 0, 0}));
    EXPECT_TRUE(VectorEquals(dense.col(2), Vector{0, 1, 0, 0}));
    EXPECT_TRUE(VectorEquals(dense.col(3), Vector{0, 0, 1, 0}));
    CheckSparseMatrixBlock(T, dense);
  }

  TEST_F(SparseMatrixTest, ArmaVariance) {
    Vector coefficients = {1, .2, .5, 0};
    NEW(ArmaStateSpaceVarianceMatrix, V)(coefficients, 1.7);
    Matrix dense = 1.7 * coefficients.outer();
    CheckSparseMatrixBlock(V, dense);
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

  Matrix LocalLinearConstraintMatrix(int dim) {
    Matrix ans(dim, dim, 0.0);
    int reduced_dim = dim / 2;
    ans.diag() += 1.0;

    Vector J(dim);
    for (int i = 0; i < reduced_dim; ++i) {
      J[2 * i] = 1.0;
    }
    ans.add_outer(J, J, -1.0 / reduced_dim);
    return ans;
  }

  TEST(LltTestFunction, MatchesDenseMatrix) {
    Matrix Expected(
        std::vector<Vector>{
          Vector{.5, 0, -.5, 0},
          Vector{0, 1, 0, 0},
          Vector{-.5, 0, .5, 0},
          Vector{0, 0, 0, 1}
        });
    Matrix observed = LocalLinearConstraintMatrix(4);
    EXPECT_TRUE(MatrixEquals(Expected, observed));
  }

  TEST_F(SparseMatrixTest, SubsetEffectConstraintMatrixTest) {
    Matrix dense = LocalLinearConstraintMatrix(8);
    NEW(SubsetEffectConstraintMatrix, sparse)(8, 2);
    CheckSparseMatrixBlock(sparse, dense);
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

  TEST_F(SparseMatrixTest, StackedRegressionCoefficientsTest) {
    NEW(StackedRegressionCoefficients, sparse)();
    std::vector<Ptr<GlmCoefs>> coefficients;
    for (int i = 0; i < 3; ++i) {
      NEW(GlmCoefs, row)(rnorm_vector(5, 3.0, 1.0));
      coefficients.push_back(row);
      sparse->add_row(row);
    }
    Matrix dense = sparse->dense();
    EXPECT_EQ(dense.nrow(), 3);
    EXPECT_EQ(dense.ncol(), 5);
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 5; ++j) {
        EXPECT_DOUBLE_EQ(dense(i, j), coefficients[i]->value()[j]);
      }
    }
    CheckSparseKalmanMatrix(*sparse);
    CheckSparseMatrixBlock(sparse, dense);
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

  TEST_F(SparseMatrixTest, StackedMatrixBlockTest) {
    StackedMatrixBlock tall;
    Matrix tall_dense(6, 2);
    tall_dense.randomize();
    tall.add_block(new DenseMatrix(tall_dense));
    CheckSparseKalmanMatrix(tall);

    StackedMatrixBlock square;
    SpdMatrix square_dense(2);
    square_dense.randomize();
    square.add_block(new DenseMatrix(square_dense));
    CheckSparseKalmanMatrix(square);

    // Check that things work okay with multiple matrices in the stack.
    tall.add_block(new DenseMatrix(square_dense));
    tall.add_block(new DenseMatrix(tall_dense));
    EXPECT_EQ(6 + 2 + 6, tall.nrow());
    EXPECT_EQ(2, tall.ncol());
    CheckSparseKalmanMatrix(tall);
  }

  TEST_F(SparseMatrixTest, StackedRegressionCoefficients) {
    std::vector<Ptr<GlmCoefs>> beta;
    for (int i = 0; i < 6; ++i) {
      beta.push_back(new GlmCoefs(rnorm_vector(4, 0.0, 1.0)));
    }

    StackedRegressionCoefficients sparse;
    for (int i = 0; i < beta.size(); ++i) {
      sparse.add_row(beta[i]);
    }
    EXPECT_EQ(sparse.nrow(), beta.size());
    EXPECT_EQ(sparse.ncol(), 4);

    // Check the matrix when everything is included.
    CheckSparseKalmanMatrix(sparse);

    // Now drop a few elements and make sure everything still works.
    beta[0]->drop(1);
    beta[1]->drop_all();
    Matrix dense = sparse.dense();

    // Check that the dense matrix is as expected.
    Matrix manual_dense(sparse.nrow(), sparse.ncol());
    for (int i = 0; i < beta.size(); ++i) {
      manual_dense.row(i) = beta[i]->Beta();
    }
    EXPECT_TRUE(MatrixEquals(manual_dense, dense));
    CheckSparseKalmanMatrix(sparse);
  }

  // Test the transition matrix from the Harvey cumulator in
  // AggregatedStateSpaceRegression.
  TEST_F(SparseMatrixTest, AccumulatorTransitionMatrixTest) {
    NEW(SeasonalStateModel, seasonal_model)(4);
    BlockDiagonalMatrix transition;
    transition.add_block(seasonal_model->state_transition_matrix(3));
    AccumulatorTransitionMatrix sparse(
        &transition, seasonal_model->observation_matrix(3), 1.0, true);
    CheckSparseKalmanMatrix(sparse);
  }

  // Test the state variance matrix from the Harvey cumulator in
  // AggregatedStateSpaceRegression.
  TEST_F(SparseMatrixTest, AccumulatorStateVarianceMatrixTest) {
    NEW(SeasonalStateModel, seasonal_model)(4);
    BlockDiagonalMatrix RQR;
    RQR.add_block(seasonal_model->state_variance_matrix(7));
    AccumulatorStateVarianceMatrix V(
        &RQR, seasonal_model->observation_matrix(7), 1.2);
    CheckSparseKalmanMatrix(V);
  }

  // Test a row expander matrix where some components have null dimension.
  // E.g. a static regression model, or a seasonal model for a time period that
  // does not begin a new season.
  TEST_F(SparseMatrixTest, NullColumns) {
    NEW(LocalLinearTrendMatrix, m1)();
    NEW(NullMatrix, m2)(4);

    NEW(ErrorExpanderMatrix, m);
    m->add_block(m1);
    m->add_block(m2);

    Matrix dense = rbind(m1->dense(), Matrix(4, 2, 0.0));
    EXPECT_TRUE(MatrixEquals(dense, m->dense()));

    CheckSparseKalmanMatrix(*m);
  }

  TEST_F(SparseMatrixTest, DenseSparseRankOneTest) {
    Vector dense(4);
    dense.randomize();

    SparseVector sv(8);
    sv[0] = 1.8;
    sv[5] = 4.2;

    DenseSparseRankOneMatrixBlock sparse(dense, sv);
    CheckSparseKalmanMatrix(sparse);
  }

  Matrix random_dense_matrix(int nrow, int ncol) {
    Matrix ans(nrow, ncol);
    ans.randomize();
    return ans;
  }

  TEST_F(SparseMatrixTest, MatrixProductTest) {
    SparseMatrixProduct sparse;
    NEW(DenseMatrix, m1)(random_dense_matrix(3,4));
    NEW(DenseMatrix, m2)(random_dense_matrix(4, 2));
    NEW(DenseMatrix, m3)(random_dense_matrix(3, 2));

    sparse.add_term(m1);
    sparse.add_term(m2);
    sparse.add_term(m3, true);
    CheckSparseKalmanMatrix(sparse);
  }

  TEST_F(SparseMatrixTest, SparseMatrixSumTest) {
    SparseMatrixSum sparse;

    NEW(DenseMatrix, m1)(random_dense_matrix(3, 4));
    NEW(DenseMatrix, m2)(random_dense_matrix(3, 4));
    NEW(DenseMatrix, m3)(random_dense_matrix(3, 4));

    sparse.add_term(m1);
    sparse.add_term(m2);
    sparse.add_term(m3, -1);

    Matrix dense = m1->dense() + m2->dense() - m3->dense();
    EXPECT_TRUE(MatrixEquals(dense, sparse.dense()))
        << "dense = \n" << dense
        << "dense() = \n" << sparse.dense();

    CheckSparseKalmanMatrix(sparse);
  }

  TEST_F(SparseMatrixTest, SparseBinomialInverseTest) {
    SpdMatrix A(4);
    A.randomize();
    SpdMatrix B(2);
    B.randomize();
    Matrix U(4, 2);
    U.randomize();

    NEW(DenseSpd, Ainv)(A.inv());
    NEW(DenseSpd, SparseB)(B);
    NEW(DenseMatrix, SparseU)(U);

    SpdMatrix M = A + U * B * U.transpose();
    SpdMatrix dense = M.inv();
    SpdMatrix Ainv_dense = Ainv->value();
    double Ainv_logdet = Ainv_dense.logdet();
    SparseBinomialInverse sparse(Ainv, SparseU, B, Ainv_logdet);

    EXPECT_NEAR(dense.logdet(), sparse.logdet(), 1e-5);

    EXPECT_TRUE(MatrixEquals(dense, sparse.dense()))
        << "dense = \n" << dense
        << "dense() = \n" << sparse.dense();

    CheckSparseKalmanMatrix(sparse);
  }

  TEST_F(SparseMatrixTest, WoodburyTest) {
    SpdMatrix A(4);
    A.randomize();
    SpdMatrix B(2);
    B.randomize();
    Matrix U(4, 2);
    U.randomize();

    SpdMatrix Ainv_dense = A.inv();
    double Ainv_logdet = Ainv_dense.logdet();
    NEW(DenseSpd, Ainv)(Ainv_dense);
    Ptr<DenseMatrix> SparseU(new DenseMatrix(U));

    SparseWoodburyInverse woody(Ainv, Ainv_logdet, SparseU, B.inv());

    SpdMatrix M1 = A + U * B * U.transpose();
    SpdMatrix dense = M1.inv();

    EXPECT_TRUE(MatrixEquals(woody.dense(), dense));
    EXPECT_NEAR(dense.logdet(), woody.logdet(), 1e-6);
    SpdMatrix Id(4, 1.0);
    EXPECT_TRUE(MatrixEquals(woody * M1, Id));
    EXPECT_TRUE(MatrixEquals(dense * M1, Id));
    CheckSparseKalmanMatrix(woody);
  }

}  // namespace

#include "gtest/gtest.h"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "distributions.hpp"
#include "cpputil/math_utils.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class DiagonalMatrixTest : public ::testing::Test {
   protected:
    DiagonalMatrixTest() {
      GlobalRng::rng.seed(8675309);
    }

    Matrix DenseMatrix(const DiagonalMatrix &diagonal) {
      Matrix ans(diagonal.nrow(), diagonal.ncol(), 0.0);
      ans.diag() = diagonal.diag();
      return ans;
    }
  };

  TEST_F(DiagonalMatrixTest, Constructors) {
    DiagonalMatrix empty;
    EXPECT_EQ(0, empty.nrow());
    EXPECT_EQ(0, empty.ncol());

    DiagonalMatrix single(1);
    EXPECT_EQ(1, single.nrow());
    EXPECT_EQ(1, single.ncol());
    EXPECT_DOUBLE_EQ(0.0, single.diag()[0]);
  }

  TEST_F(DiagonalMatrixTest, MultiplyMatrix) {
    DiagonalMatrix diagonal(3);
    diagonal.randomize();
    Matrix dense = DenseMatrix(diagonal);
    
    Matrix foo(3, 3);
    foo.randomize();
    EXPECT_TRUE(MatrixEquals(diagonal * foo, dense * foo));
    EXPECT_TRUE(MatrixEquals(diagonal.Tmult(foo), dense.Tmult(foo)));

    EXPECT_TRUE(MatrixEquals(foo * diagonal, foo * dense));
    EXPECT_TRUE(MatrixEquals(foo.Tmult(diagonal), foo.Tmult(dense)));
  }

  TEST_F(DiagonalMatrixTest, SpdMultiply) {
    DiagonalMatrix diagonal(3);
    diagonal.randomize();
    Matrix dense = DenseMatrix(diagonal);

    SpdMatrix foo(3);
    foo.randomize();
    EXPECT_TRUE(MatrixEquals(diagonal * foo, dense * foo));
    EXPECT_TRUE(MatrixEquals(foo * diagonal, foo * dense));

    EXPECT_TRUE(MatrixEquals(diagonal.sandwich(foo), sandwich(dense, foo)));
  }

  TEST_F(DiagonalMatrixTest, VectorMultiply) {
    DiagonalMatrix diagonal(3);
    diagonal.randomize();

    Vector v(3);
    v.randomize();
    EXPECT_TRUE(VectorEquals(diagonal * v, DenseMatrix(diagonal) * v));
    EXPECT_TRUE(VectorEquals(v * diagonal, v * DenseMatrix(diagonal)));

    VectorView view(v);
    EXPECT_TRUE(VectorEquals(diagonal * view, DenseMatrix(diagonal) * view));
    EXPECT_TRUE(VectorEquals(view *diagonal, view * DenseMatrix(diagonal)));

    ConstVectorView cview(v);
    EXPECT_TRUE(VectorEquals(diagonal * cview, DenseMatrix(diagonal) * cview));
    EXPECT_TRUE(VectorEquals(cview *diagonal, cview * DenseMatrix(diagonal)));

    Vector vcopy = v;
    diagonal.multiply_inplace(v);
    EXPECT_TRUE(VectorEquals(v, diagonal * vcopy));
  }

  TEST_F(DiagonalMatrixTest, InverseEtc) {
    DiagonalMatrix diagonal(3);
    diagonal.randomize();
    Matrix dense = DenseMatrix(diagonal);

    EXPECT_TRUE(MatrixEquals(diagonal.inv(), dense.inv()));
    EXPECT_TRUE(MatrixEquals(diagonal.inner(), dense.inner()));

    Vector v(3);
    v.randomize();
    EXPECT_TRUE(VectorEquals(diagonal.solve(v), dense.solve(v)));

    EXPECT_DOUBLE_EQ(diagonal.det(), dense.det());
    EXPECT_DOUBLE_EQ(diagonal.logdet(), dense.logdet());
  }
  
  
}  // namespace

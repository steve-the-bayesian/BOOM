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
}  // namespace

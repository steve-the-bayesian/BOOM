#include "gtest/gtest.h"
#include "distributions.hpp"
#include "LinAlg/SVD.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class SVDTest : public ::testing::Test {
   protected:
    SVDTest()
    {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(SVDTest, WholeClass) {
    SpdMatrix spd(4);
    spd.randomize();

    SingularValueDecomposition svd_square(spd);

    EXPECT_TRUE(MatrixEquals(spd, svd_square.original_matrix()));

    EXPECT_TRUE(MatrixEquals(spd,
                             svd_square.left() * DiagonalMatrix(svd_square.values())
                             * svd_square.right().transpose()));

    EXPECT_TRUE(MatrixEquals(spd.inv(),
                             svd_square.inv()));

    Matrix m(4, 4);
    m.randomize();
    EXPECT_TRUE(MatrixEquals(
        svd_square.solve(m),
        spd.solve(m)));

    Matrix rectangle(5, 2);
    rectangle.randomize();
    SingularValueDecomposition svd_rect(rectangle);
    EXPECT_EQ(svd_rect.values().size(), 2);
    EXPECT_EQ(svd_rect.left().nrow(), 5);
    EXPECT_EQ(svd_rect.left().ncol(), 2);
    EXPECT_EQ(svd_rect.right().nrow(), 2);
    EXPECT_EQ(svd_rect.right().ncol(), 2);
    EXPECT_TRUE(MatrixEquals(
        rectangle, svd_rect.original_matrix()));
    EXPECT_TRUE(MatrixEquals(
        rectangle,
        svd_rect.left() * DiagonalMatrix(svd_rect.values())
        * svd_rect.right().transpose()));

  }
  
}  // namespace

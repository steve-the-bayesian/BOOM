#include "gtest/gtest.h"
#include "Models/MatrixNormalModel.hpp"
#include "Models/MvnModel.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MatrixNormalTest : public ::testing::Test {
   protected:
    MatrixNormalTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MatrixNormalTest, Dimensions) {
    MatrixNormalModel model(3, 4);
    EXPECT_EQ(3, model.nrow());
    EXPECT_EQ(4, model.ncol());

    EXPECT_EQ(3, model.row_variance().nrow());
    EXPECT_EQ(3, model.row_variance().ncol());
    EXPECT_EQ(3, model.row_precision().nrow());
    EXPECT_EQ(3, model.row_precision().ncol());
  }

  TEST_F(MatrixNormalTest, CompareToMvn) {
    SpdMatrix row_variance(2);
    row_variance.randomize();
    SpdMatrix col_variance(3);
    col_variance.randomize();
    Matrix mu(2, 3);
    mu.randomize();

    SpdMatrix variance = Kronecker(col_variance, row_variance);
    MvnModel mvn(vec(mu), variance);

    MatrixNormalModel model(mu, row_variance, col_variance);
    
    int ndraws = 10000;
    Matrix mvn_draws(ndraws, 2 * 3);
    Matrix draws(ndraws, 2 * 3);
    for (int i = 0; i < ndraws; ++i) {
      mvn_draws.row(i) = mvn.sim();
      draws.row(i) = vec(model.simulate());
    }

    for (int i = 0; i < 2 * 3; ++i) {
      EXPECT_TRUE(TwoSampleKs(draws.col(i), mvn_draws.col(i)));
    }

    EXPECT_NEAR(model.logp(mvn_draws.row(0)),
                mvn.logp(mvn_draws.row(0)),
                1e-5);
  }

  
}  // namespace

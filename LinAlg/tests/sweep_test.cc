#include "gtest/gtest.h"
#include "distributions.hpp"
#include "LinAlg/SWEEP.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class SweepTest : public ::testing::Test {
   protected:
    SweepTest()
        : spd_(4)
    {
      GlobalRng::rng.seed(8675309);
      spd_.randomize();
    }
    SpdMatrix spd_;
  };

  TEST_F(SweepTest, Construction) {
    SweptVarianceMatrix swp(spd_);
    swp.SWP(2);
    EXPECT_FALSE(MatrixEquals(swp.swept_matrix(), spd_));

    swp.RSW(2);
    EXPECT_TRUE(MatrixEquals(swp.swept_matrix(), spd_));
  }

  TEST_F(SweepTest, MultipleCallsAreANoOp) {
    SweptVarianceMatrix swp(spd_);
    swp.SWP(1);
    Matrix mat = swp.swept_matrix();

    swp.SWP(1);
    EXPECT_TRUE(MatrixEquals(mat, swp.swept_matrix()));

    // Index 2 is currently unswept, so RSW shouldn't do anything.
    swp.RSW(2);
    EXPECT_TRUE(MatrixEquals(mat, swp.swept_matrix()));
  }

  TEST_F(SweepTest, GeneratesInverse) {
    SweptVarianceMatrix swp(spd_);
    for (int i = 0; i < spd_.nrow(); ++i) {
      swp.SWP(i);
    }
    SpdMatrix foo = swp.swept_matrix() * spd_;
    EXPECT_TRUE(MatrixEquals(foo, SpdMatrix(spd_.nrow(), -1.0)));
  }

  TEST_F(SweepTest, RegressionTest) {
    SweptVarianceMatrix swp(spd_);

    swp.SWP(1);
    swp.SWP(2);
    swp.SWP(3);

    EXPECT_EQ(1, swp.Beta().nrow());
    EXPECT_EQ(3, swp.Beta().ncol());

    Vector mean = {1.2, 2.8, -4.3, 0.7};

    SubMatrix Sigma_22(spd_, 1, 3, 1, 3);
    SubMatrix Sigma_12(spd_, 0, 0, 1, 3);

    Matrix coef = Sigma_22.to_matrix().solve(Sigma_12.to_matrix().transpose());
    EXPECT_TRUE(MatrixEquals(coef.transpose(), swp.Beta()));

    Vector y  = {8, 6, 7, 5};
    double conditional_mean =
        (coef.transpose() * (ConstVectorView(y, 1) - ConstVectorView(mean, 1)))[0]
        + 1.2;

    Vector yhat = swp.conditional_mean(ConstVectorView(y, 1), mean);
    EXPECT_NEAR(yhat[0], conditional_mean, 1e-4);
  }

  TEST_F(SweepTest, VarianceTest) {
    SweptVarianceMatrix swp(spd_);

    swp.SWP(1);
    swp.SWP(2);
    swp.SWP(3);

    // V = Sigma22 - Sigma21 * Sigma11^-1 Sigma12

    Matrix Sigma12 = SubMatrix(spd_, 0, 0, 1, 3).to_matrix();
    Matrix Sigma21 = Sigma12.transpose();
    SpdMatrix Sigma22 = SubMatrix(spd_, 1, 3, 1, 3).to_matrix();

    double residual_variance =
        spd_(0, 0) - (Sigma12 * Sigma22.inv() * Sigma21)(0, 0);
    EXPECT_NEAR(residual_variance, swp.residual_variance()(0, 0), 1e-4);

  }

}  // namespace

#include "gtest/gtest.h"

#include "Models/GP/GaussianProcessRegressionModel.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class KernelTest : public ::testing::Test {
   protected:
    KernelTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(KernelTest, RadialBasisTest) {
    RadialBasisFunction k(1.7);

    Vector x1(3);
    Vector x2(3);
    x1.randomize();
    x2.randomize();

    EXPECT_DOUBLE_EQ(k(x1, x2),
                     exp(-2 * (x1 - x2).normsq() / 1.7 / 1.7));

  }

  TEST_F(KernelTest, MahalanobisKernelTest) {
    int dim = 3;
    int sample_size = 10;
    Matrix X(sample_size, dim);
    X.randomize();

    double scale = 1.3;
    double dshrink = 0.05;
    MahalanobisKernel k(X, scale, dshrink);

    SpdMatrix xtx = scale * X.transpose() * X / sample_size;
    SpdMatrix shrunk = self_diagonal_average(xtx, dshrink);

    SpdMatrix xtx_inv = shrunk.inv();
    SpdMatrix M = xtx_inv;

    Vector x1(dim);
    x1.randomize();
    Vector x2(dim);
    x2.randomize();

    EXPECT_NEAR(k(x1, x2), exp(-.5 * M.Mdist(x1, x2)), 1e-8)
        << "M = \n" << M
        << "kernel = \n"
        << k;
  }

  // Check that setting the scale works as expected.
  TEST_F(KernelTest, MahalanobisKernelScaleTest) {
    int dim = 3;
    int sample_size = 10;
    Matrix X(sample_size, dim);
    X.randomize();

    double scale = 1.3;
    double dshrink = 0.05;
    MahalanobisKernel k1(X, scale, dshrink);

    // k2 is k1 with a different scale parameter.
    MahalanobisKernel k2(X, 1.0, dshrink);

    Vector x1(dim);
    x1.randomize();
    Vector x2(dim);
    x2.randomize();

    // The different scale parameters give different kernel values.
    EXPECT_GT(fabs(k1(x1, x2) - k2(x1, x2)), 1e-2);

    // Setting the scale on k2 now gives the same kernel values.
    k2.set_scale(scale);
    EXPECT_NEAR(k1(x1, x2), k2(x1, x2), 1e-8);
  }


}  // namespace

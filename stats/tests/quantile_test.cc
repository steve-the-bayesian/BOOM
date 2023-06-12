#include "gtest/gtest.h"
#include "stats/quantile.hpp"

#include "LinAlg/Vector.hpp"
#include "distributions.hpp"

#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class QuantileTest : public ::testing::Test {
   protected:
    QuantileTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // Check that 'quantile' called on a vector of length 1 gives the single
  // element.
  TEST_F(QuantileTest, Singletons) {
    Vector zero{0.0};
    EXPECT_NEAR(0.0, quantile(zero, 0.0), 1e-8);
    EXPECT_NEAR(0.0, quantile(zero, 0.5), 1e-8);
    EXPECT_NEAR(0.0, quantile(zero, 0.7), 1e-8);
    EXPECT_NEAR(0.0, quantile(zero, 1.0), 1e-8);

    Vector one{1.0};
    EXPECT_NEAR(1.0, quantile(one, 0.0), 1e-8);
    EXPECT_NEAR(1.0, quantile(one, 0.5), 1e-8);
    EXPECT_NEAR(1.0, quantile(one, 0.7), 1e-8);
    EXPECT_NEAR(1.0, quantile(one, 1.0), 1e-8);
  }

  TEST_F(QuantileTest, ScalarQuantiles) {
    Vector y = {1.0, 3.0, 7.0};
    Vector z = {1, 3, 10, 12};

    EXPECT_NEAR(3.0, quantile(y, 0.5), 1e-8);

    // The 0.5 quantile is 3.0.  The 1.0 quantile is 7.0.  60% is 20% between 3.0 and 7.0.
    EXPECT_NEAR(.8 * 3.0 + .2 * 7, quantile(y, 0.6), 1e-8);

    EXPECT_NEAR(6.5, median(z), 1e-8);
  }

  TEST_F(QuantileTest, VectorQuantiles) {
    Matrix data(50, 4);
    data.randomize();

    double target_quantile = 0.37;
    Vector quantiles = quantile(data, target_quantile);
    Vector direct_quantiles(data.ncol());
    for (int i = 0; i < data.ncol(); ++i) {
      direct_quantiles[i] = quantile(data.col(i), target_quantile);
    }
    EXPECT_TRUE(VectorEquals(direct_quantiles, quantiles));


    Vector scalar_data(100);
    Vector target_quantiles = {0.0, 0.2, 0.5, 1.0};
    quantiles = quantile(scalar_data, target_quantiles);
    direct_quantiles.resize(target_quantiles.size());
    for (int i = 0; i < direct_quantiles.size(); ++i) {
      direct_quantiles[i] = quantile(scalar_data, target_quantiles[i]);
    }
    EXPECT_TRUE(VectorEquals(direct_quantiles, quantiles));
  }

  TEST_F(QuantileTest, MatrixQuantiles) {
    Matrix data(100, 4);
    data.randomize();
    Vector target_quantiles = {0.0, 0.2, 0.5, 0.8, 0.93};

    Matrix quantiles = quantile(data, target_quantiles);
    EXPECT_EQ(quantiles.ncol(), data.ncol());
    EXPECT_EQ(quantiles.nrow(), target_quantiles.size());

    for (int i = 0; i < data.ncol(); ++i) {
      EXPECT_TRUE(VectorEquals(quantiles.col(i),
                               quantile(data.col(i), target_quantiles)))
          << "Column " << i << "didn't work.\n"
          << cbind(quantiles.col(i), quantile(data.col(i), target_quantiles));
    }
  }

}  // namespace

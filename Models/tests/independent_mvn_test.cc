#include "gtest/gtest.h"
#include "Models/IndependentMvnModel.hpp"
#include "distributions.hpp"
#include "cpputil/lse.hpp"
#include "numopt/NumericalDerivatives.hpp"
#include "numopt/Integral.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class IndependentMvnTest : public ::testing::Test {
   protected:
    IndependentMvnTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(IndependentMvnTest, Suf) {
    int dim = 3;

    IndependentMvnSuf suf(dim);
    for (int i = 0; i < dim; ++i) {
      EXPECT_DOUBLE_EQ(0, suf.n(i));
      EXPECT_DOUBLE_EQ(0, suf.sum(i));
      EXPECT_DOUBLE_EQ(0, suf.ybar(i));
      EXPECT_DOUBLE_EQ(0, suf.sumsq(i));
      EXPECT_DOUBLE_EQ(0, suf.sample_var(i));
    }

    Vector mean = {-2, 2, 48};
    DiagonalMatrix variance(Vector{1, 2, 3});
    Vector y = rmvn(mean, variance);
    suf.update_raw(y);
    for (int i = 0; i < dim; ++i) {
      EXPECT_DOUBLE_EQ(1, suf.n(i));
      EXPECT_DOUBLE_EQ(y[i], suf.sum(i));
      EXPECT_DOUBLE_EQ(y[i], suf.ybar(i));
      EXPECT_DOUBLE_EQ(square(y[i]), suf.sumsq(i));
      EXPECT_DOUBLE_EQ(0, suf.sample_var(i));
    }

    Vector z = rmvn(mean, variance);
    suf.update_raw(z);
    for (int i = 0; i < dim; ++i) {
      EXPECT_DOUBLE_EQ(2, suf.n(i));
      EXPECT_DOUBLE_EQ(y[i] + z[i], suf.sum(i));
      EXPECT_DOUBLE_EQ(.5 * (y[i] + z[i]), suf.ybar(i));
      EXPECT_DOUBLE_EQ(square(y[i]) + square(z[i]), suf.sumsq(i));
      EXPECT_NEAR(
          square(y[i] - suf.ybar(i)) + square(z[i] - suf.ybar(i)),
          suf.sample_var(i),
          1e-5);
      EXPECT_NEAR(suf.centered_sumsq(i, suf.ybar(i)), suf.sample_var(i), 1e-5);
    }

    Vector vsuf = suf.vectorize();
    EXPECT_EQ(9, vsuf.size());
    EXPECT_EQ(vsuf[0], suf.n(0));
    EXPECT_EQ(vsuf[1], suf.sum(0));
    EXPECT_EQ(vsuf[2], suf.sumsq(0));
    EXPECT_EQ(vsuf[3], suf.n(1));
    EXPECT_EQ(vsuf[4], suf.sum(1));
    EXPECT_EQ(vsuf[5], suf.sumsq(1));
    EXPECT_EQ(vsuf[6], suf.n(2));
    EXPECT_EQ(vsuf[7], suf.sum(2));
    EXPECT_EQ(vsuf[8], suf.sumsq(2));
  }
  
}  // namespace

#include "gtest/gtest.h"

#include "distributions.hpp"
#include "LinAlg/Vector.hpp"
#include "stats/EmpiricalDensity.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;

  class EmpiricalDensityTest : public ::testing::Test {
   protected:
    EmpiricalDensityTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(EmpiricalDensityTest, NormalDistributionLargeSample) {
    Vector draws(10000);
    for (int i = 0; i < draws.size(); ++i) {
      draws[i] = rnorm(3, 7);
    }
    EmpiricalDensity density(draws);
    double err = 0;
    Matrix eval(draws.size(), 3);
    eval.col(0) = sort(draws);
    for (int i = 1; i < nrow(eval); ++i) {
      eval(i, 1) = density(eval(i, 0));
      eval(i, 2) = dnorm(eval(i, 0), 3, 7);
      double dx = eval(i, 0) - eval(i - 1, 0);
      err += fabs(eval(i, 1) - eval(i, 2)) * dx;
    }
    EXPECT_LT(err, .05);
  }

  TEST_F(EmpiricalDensityTest, Exponential) {
    Vector draws(10000);
    for (int i = 0; i < draws.size(); ++i) {
      draws[i] = rexp(3);
    }
    EmpiricalDensity density(draws);
    double err = 0;
    Matrix eval(draws.size(), 3);
    eval.col(0) = sort(draws);
    for (int i = 1; i < nrow(eval); ++i) {
      eval(i, 1) = density(eval(i, 0));
      eval(i, 2) = dexp(eval(i, 0), 3);
      double dx = eval(i, 0) - eval(i-1, 0);
      err += fabs(eval(i, 1) - eval(i, 2)) * dx;
    }
    EXPECT_LT(err, .05);
  }
  
}  // namespace

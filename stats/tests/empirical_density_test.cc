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
    double max_density = dnorm(3, 3, 7);

    double err = 0;
    Matrix eval(draws.size(), 3);
    eval.col(0) = sort(draws);
    for (int i = 0; i < nrow(eval); ++i) {
      eval(i, 1) = density(eval(i, 0));
      eval(i, 2) = dnorm(eval(i, 0), 3, 7);
      err = std::max(err, fabs(eval(i, 2) - eval(i, 1)));
    }
    EXPECT_LT(err / max_density, .03) << eval;
  }

  TEST_F(EmpiricalDensityTest, Exponential) {
    Vector draws(10000);
    for (int i = 0; i < draws.size(); ++i) {
      draws[i] = rexp(3);
    }
    EmpiricalDensity density(draws);
    double max_density = dexp(0, 3);

    double err = 0;
    Matrix eval(draws.size(), 3);
    eval.col(0) = sort(draws);
    for (int i = 0; i < nrow(eval); ++i) {
      eval(i, 1) = density(eval(i, 0));
      eval(i, 2) = dexp(eval(i, 0), 3);
      err = std::max(err, fabs(eval(i, 2) - eval(i, 1)));
    }
    EXPECT_LT(err / max_density, .05) << eval;
  }
  
}  // namespace

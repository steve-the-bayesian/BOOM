#include "gtest/gtest.h"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include "numopt/NumericalDerivatives.hpp"
#include "stats/ECDF.hpp"
#include <functional>

namespace {

  using namespace BOOM;
  using std::cout;
  using std::endl;
  double eps = 1e-5;

  TEST(pexv, density_matches_cdf) {
    double mu = 3;
    double sigma = 7;
    std::function<double(double)> cdf = [mu, sigma](double x) {
      return pexv(x, mu, sigma);
    };
    ScalarNumericalDerivatives numeric(cdf);

    EXPECT_NEAR(numeric.first_derivative(2.8),
                dexv(2.8, mu, sigma),
                eps);
    EXPECT_NEAR(numeric.first_derivative(-2.8),
                dexv(-2.8, mu, sigma),
                eps);
    EXPECT_NEAR(numeric.first_derivative(.6),
                dexv(.6, mu, sigma),
                eps);
  }

  TEST(pexv, logscale) {
    EXPECT_NEAR(pexv(2.8, 3, 7, true),
                log(pexv(2.8, 3, 7, false)),
                eps);
  }

  TEST(cexv, logscale) {
    EXPECT_NEAR(dexv(2.8, 3, 7, true),
                log(dexv(2.8, 3, 7, false)),
                eps);
  }
  
  TEST(rexv, works_as_intended) {
    GlobalRng::rng.seed(8675309);
    int n = 100000;
    Vector y(n);
    for (int i = 0; i < n; ++i) {
      y[i] = rexv(3, 7);
    }
    EXPECT_TRUE(DistributionsMatch(y,
                                   [](double x) {return pexv(x, 3, 7);},
                                   .05))
        << "Mismatch between empirical and mathematical CDF for 'pexv' and 'rexv'.";
  }
  
}

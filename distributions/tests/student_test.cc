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

  TEST(pstudent_test, density_matches_cdf) {
    double mu = 3;
    double sigma = 7;
    double nu = 11.0;
    std::function<double(double)> cdf = [mu, sigma, nu](double x) {
      return pstudent(x, mu, sigma, nu);
    };
    ScalarNumericalDerivatives numeric(cdf);

    EXPECT_NEAR(numeric.first_derivative(2.8),
                dstudent(2.8, mu, sigma, nu),
                eps);
    EXPECT_NEAR(numeric.first_derivative(-2.8),
                dstudent(-2.8, mu, sigma, nu),
                eps);
    EXPECT_NEAR(numeric.first_derivative(.6),
                dstudent(.6, mu, sigma, nu),
                eps);
  }

  TEST(pstudent_test, logscale) {
    EXPECT_NEAR(pstudent(2.8, 3, 7, 11, true, true),
                log(pstudent(2.8, 3, 7, 11, true, false)),
                eps);
  }

  TEST(dstudent_test, logscale) {
    EXPECT_NEAR(dstudent(2.8, 3, 7, 11, true),
                log(dstudent(2.8, 3, 7, 11, false)),
                eps);
  }

  TEST(dstudent_test, works_as_intended) {
    GlobalRng::rng.seed(8675309);
    int n = 100000;
    Vector y(n);
    for (int i = 0; i < n; ++i) {
      y[i] = rstudent(3, 7, 11.0);
    }
    EXPECT_TRUE(DistributionsMatch(y,
                                   [](double x) {return pstudent(x, 3, 7, 11.0);},
                                   .05))
        << "Mismatch between empirical and mathematical CDF for 'pstudent' and 'rstudent'.";
  }

}

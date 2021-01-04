#include "gtest/gtest.h"
#include "distributions/trun_gamma.hpp"
#include <iostream>

namespace {
  using namespace BOOM;
  using std::cout;
  using std::endl;

  TEST(TrunGammaTest, Density) {
    double a = 1, b = 1e-8;
    double cut = a/b + 1;
    double x = cut + 1;
    EXPECT_NEAR(dtrun_gamma(x, a, b, cut, true, true),
                dgamma(x, a, b, true) - pgamma(cut, a, b, false, true),
                1e-3);

    EXPECT_NEAR(dtrun_gamma(x, a, b, cut, false, true),
                exp(dgamma(x, a, b, true) - pgamma(cut, a, b, false, true)),
                1e-3);

    a = 1e-3;
    b = 12;
    cut = a / b + 1;
    x = cut + 1;
    EXPECT_NEAR(dtrun_gamma(x, a, b, cut, true, true),
                dgamma(x, a, b, true) - pgamma(cut, a, b, false, true),
                1e-3);
  }

  TEST(TrunGammaTest, Simulation) {
    double a = 0.505;
    double b = 2.4091261115882602e-05;
    double cut = 11.231845316976706;
    int n = 1000;
    Vector draws(n);
    for (int i = 0; i < n; ++i) {
      draws[i] = rtrun_gamma_mt(GlobalRng::rng, a, b, cut);
    }
  }

  TEST(TrunGammaTest, negative_alpha) {
    double a = -3.8;
    double b = 1.2;
    double cut = 1e-10;
    int n = 10000;
    Vector draws(n);
    for (int i = 0; i < n; ++i) {
      draws[i] = rtrun_gamma_mt(GlobalRng::rng, a, b, cut);
    }
  }

}  // namespace

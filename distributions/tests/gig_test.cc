#include <fstream>
#include "gtest/gtest.h"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include "stats/moments.hpp"
#include "numopt/Integral.hpp"

namespace {

  using namespace BOOM;
  using std::cout;
  using std::endl;

  TEST(GigTest, Simulation) {
    GlobalRng::rng.seed(8675309);
    int niter = 10000;
    Vector draws(niter);
    double lambda = 1.0;
    double psi = 2.0;
    double chi = 3.0;

    for (int i = 0; i < niter; ++i) {
      draws[i] = rgig_mt(GlobalRng::rng, lambda, psi, chi);
    }
    double sd = BOOM::sd(draws);
    EXPECT_NEAR(mean(draws), gig_mean(lambda, psi, chi), 3 * sd / sqrt(niter));

    // std::ofstream draws_file("gig_draws.out");
    // draws_file << draws;
    // double min_draw = min(draws);
    // double max_draw = max(draws);
    // double range = max_draw - min_draw;

    // std::ofstream density_file("gig_density.out");
    // for (double x = min_draw; x <= max_draw; x += range / 20p0) {
    //   density_file << x << " " << dgig(x, lambda, psi, chi, false) << "\n";
    // }
    EXPECT_TRUE(DistributionsMatch(
        draws,
        CumulativeDistributionFunction([lambda, psi, chi](double x) {
            return dgig(x, lambda, psi, chi, false);})));

  }

  TEST(GigTest, SimulationNegativeLambda) {
    GlobalRng::rng.seed(8675309);
    int niter = 10000;
    Vector draws(niter);
    double lambda = -87.0;
    double psi = 2.0;
    double chi = 3.0;

    for (int i = 0; i < niter; ++i) {
      draws[i] = rgig_mt(GlobalRng::rng, lambda, psi, chi);
    }
    //    double sd = BOOM::sd(draws);
    //    EXPECT_NEAR(mean(draws), gig_mean(lambda, psi, chi), 3 * sd / sqrt(niter));

    std::ofstream draws_file("gig_draws.out");
    draws_file << draws;
    double min_draw = min(draws);
    double max_draw = max(draws);
    double range = max_draw - min_draw;

    std::ofstream density_file("gig_density.out");
    for (double x = min_draw; x <= max_draw; x += range / 200) {
      density_file << x << " " << dgig(x, lambda, psi, chi, false) << "\n";
    }
    EXPECT_TRUE(DistributionsMatch(
        draws,
        CumulativeDistributionFunction([lambda, psi, chi](double x) {
            return dgig(x, lambda, psi, chi, false);})));

  }

}  // namespace

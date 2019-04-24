#include "gtest/gtest.h"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include "numopt/NumericalDerivatives.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "stats/ECDF.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include <functional>

namespace {

  using namespace BOOM;
  using std::cout;
  using std::endl;

  TEST(RmvnTest, DiagonalVsSpd) {
    Vector mu = {1.2, 3.7, 9.6};
    Vector variances = {.08, 1.9, 200.62};
    DiagonalMatrix diagonal_variance(variances);
    SpdMatrix variance_matrix(3, 0.0);
    variance_matrix.diag() = variances;
    SpdMatrix precision = variance_matrix.inv();

    int ndraws = 100000;
    Matrix draws1(ndraws, mu.size());
    Matrix draws2(ndraws, mu.size());
    Vector Mdist1(ndraws);
    Vector Mdist2(ndraws);
    
    for (int i = 0; i < ndraws; ++i) {
      draws1.row(i) = rmvn(mu, variance_matrix);
      Mdist1[i] = precision.Mdist(draws1.row(i));
      
      draws2.row(i) = rmvn(mu, diagonal_variance);
      Mdist2[i] = precision.Mdist(draws2.row(i));
    }

    EXPECT_TRUE(TwoSampleKs(draws1.col(0), draws2.col(0)))
        << "Variable 0 did not match" << endl
        << AsciiDistributionCompare(draws1.col(0), draws2.col(0));
    EXPECT_TRUE(TwoSampleKs(draws1.col(1), draws2.col(1)))
        << "Variable 1 did not match" << endl
        << AsciiDistributionCompare(draws1.col(1), draws2.col(1));
    EXPECT_TRUE(TwoSampleKs(draws1.col(2), draws2.col(2)))
        << "Variable 2 did not match" << endl
        << AsciiDistributionCompare(draws1.col(2), draws2.col(2));
    EXPECT_TRUE(TwoSampleKs(Mdist1, Mdist2))
        << "Mahalinobis distances did not match" << endl
        << AsciiDistributionCompare(Mdist1, Mdist2);
  }

  TEST(ImputeMvnTest, ImputeGivesRightDistribution) {
    GlobalRng::rng.seed(8675309);
    Vector mu(5);
    mu.randomize();
    SpdMatrix Sigma(5);
    Sigma.randomize();

    Vector y = rmvn(mu, Sigma);
    Vector z = y;
    Selector observed("11010");
    observed.fill_missing_elements(y, -99);
    z[2] = -99;
    z[4] = -99;
    EXPECT_TRUE(VectorEquals(y, z));

    // Check that impute_mvn changes the two specified values and no others.
    y = impute_mvn(y, mu, Sigma, observed);
    EXPECT_NE(y[2], -99);
    EXPECT_NE(y[4], -99);
    z[2] = y[2];
    z[4] = y[4];
    EXPECT_TRUE(VectorEquals(y, z));

    
    int niter = 10000;
    Matrix draws(niter, 5);
    Matrix original_draws(niter, 5);
    for (int i = 0; i < niter; ++i) {
      Vector y = rmvn(mu, Sigma);
      original_draws.row(i) = y;
      draws.row(i) = impute_mvn(y, mu, Sigma, observed);
    }
    EXPECT_TRUE(VectorEquals(original_draws.col(0), draws.col(0)));
    EXPECT_TRUE(VectorEquals(original_draws.col(1), draws.col(1)));
    EXPECT_FALSE(VectorEquals(original_draws.col(2), draws.col(2)));
    EXPECT_TRUE(VectorEquals(original_draws.col(3), draws.col(3)));
    EXPECT_FALSE(VectorEquals(original_draws.col(4), draws.col(4)));

    EXPECT_TRUE(TwoSampleKs(draws.col(2), original_draws.col(2)));
    EXPECT_TRUE(TwoSampleKs(draws.col(4), original_draws.col(4)));
  }

}  // namespace 

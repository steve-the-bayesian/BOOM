#include "gtest/gtest.h"
#include "distributions.hpp"
#include "test_utils/test_utils.hpp"
#include "numopt/NumericalDerivatives.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Array.hpp"
#include "stats/ECDF.hpp"
#include "stats/moments.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include <functional>

namespace {

  using namespace BOOM;
  using std::cout;
  using std::endl;

  TEST(MatrixNormalTest, Simulation) {
    GlobalRng::rng.seed(8675309);
    
    Matrix Mu(3, 4);
    Mu.randomize();
    SpdMatrix row_variance(3);
    row_variance.randomize();
    SpdMatrix row_precision = row_variance.inv();
    
    SpdMatrix col_variance(4);
    col_variance.randomize();
    SpdMatrix col_precision = col_variance.inv();

    SpdMatrix global_variance = Kronecker(col_variance, row_variance);
    int niter = 10000;
    Array draws({niter, 3, 4});
    Matrix vectorized_draws(niter, 12);
    
    for (int i = 0; i < niter; ++i) {
      Matrix draw = rmatrix_normal_ivar(Mu, row_precision, col_precision);
      draws.slice(i, -1, -1) = draw;
      vectorized_draws.row(i) = vec(draw);
    }

    // Check the mean vector.
    Vector empirical_mean = mean(vectorized_draws);
    double se = sqrt(max(diag(global_variance)) / niter);
    EXPECT_TRUE(VectorEquals(empirical_mean, vec(Mu), 4 * se));

    // Check that the marginal distributions are all correct.
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 4; ++j) {
        const std::function<double(double)> target =
            [i, j, &Mu, &row_variance, &col_variance](double x) {
          return pnorm(x, Mu(i, j), sqrt(row_variance(i, i) * col_variance(j, j)));};
        EXPECT_TRUE(DistributionsMatch(draws.vector_slice(-1, i, j), target));
      }
    }

    // Check that the variance matrix is close to the empirical variance.
    SpdMatrix empirical_variance = var(vectorized_draws);
    int error_count = 0;
    for (int i = 0; i < nrow(empirical_variance); ++i) {
      EXPECT_LE( (niter - 1) * empirical_variance(i, i) / global_variance(i, i),
                 qchisq(.95, niter - 1, true, false))
          << "i = " << i << " critical value with " << niter - 1
          << " df = " << qchisq(.95, niter - 1, true, false) << std::endl
          << "empirical_variance   = " << empirical_variance(i, i) << std::endl
          << "theoretical variance = " << global_variance(i, i) << std::endl
          << " " << ++error_count;
    }
    if (error_count > 0) {
      std::cout << "empirical_variance: " << endl << empirical_variance
                << "true variance: " << endl
                << global_variance << endl;
    }
    
  }    
    
}  // namespace 

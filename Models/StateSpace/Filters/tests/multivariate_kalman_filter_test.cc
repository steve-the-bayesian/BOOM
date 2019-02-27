#include "gtest/gtest.h"
#include "distributions.hpp"

#include "Models/StateSpace/Filters/SparseMatrix.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  
  class MultivariateKalmanFilterTest : public ::testing::Test {
   protected:
    MultivariateKalmanFilterTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // This test checks 3 thing:
  // 1) the binomial inverse theorem (BIT) used to compute Finv * error
  // 2) the matrix determinant lemma used to compute log(det(Finv))
  // 3) the computation of the kalman gain from the BIT.
  TEST_F(MultivariateKalmanFilterTest, CheckTheMath) {
    // The notation used here follows Durbin and Koopman.
    int ydim = 4;
    int state_dim = 2;
    
    // Residual variance matrix for observed data given state.
    SpdMatrix H(ydim, 0.0);
    H.diag() = pow(rnorm_vector(ydim, 0, 1), 2);
    SpdMatrix Hinv(ydim, 0.0);
    Hinv.diag() = 1.0 / H.diag();

    Matrix transition_dense(state_dim, state_dim);
    transition_dense.randomize();
    NEW(DenseMatrix, transition)(transition_dense);

    // Observation coefficients.
    Matrix Zdense(ydim, state_dim);  
    Zdense.randomize();
    NEW(DenseMatrix, Z)(Zdense);

    // State variance matrix.
    SpdMatrix P(state_dim);
    P.randomize();

    Matrix TPZprime_dense = transition_dense * P * Zdense.transpose();
    Matrix TPZprime = (*Z * (*transition * P).transpose()).transpose();
    EXPECT_TRUE(MatrixEquals(TPZprime, TPZprime_dense));

    SpdMatrix P2 = transition_dense * P * transition_dense.transpose();

    SpdMatrix P3 = P;
    transition->sandwich_inplace(P3);
    EXPECT_TRUE(MatrixEquals(P3, P2));

    Matrix kalman_gain(state_dim, ydim);
    kalman_gain.randomize();

    P2 -= TPZprime * kalman_gain.transpose();
    P3 -= TPZprime.multT(kalman_gain);
    EXPECT_TRUE(MatrixEquals(P2, P3));
  }    

}  // namespace

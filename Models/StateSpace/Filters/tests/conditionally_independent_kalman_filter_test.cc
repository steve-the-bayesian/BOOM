#include "gtest/gtest.h"
#include "distributions.hpp"

#include "Models/StateSpace/StateSpaceModel.hpp"
#include "Models/StateSpace/StateModels/LocalLevelStateModel.hpp"
#include "Models/StateSpace/MultivariateStateSpaceModel.hpp"


#include "LinAlg/DiagonalMatrix.hpp"
#include "LinAlg/LU.hpp"
#include "LinAlg/Cholesky.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;
  using Marginal = Kalman::ConditionallyIndependentMarginalDistribution;

  inline bool is_pos_def(const SpdMatrix &v) {
    Cholesky chol(v);
    return chol.is_pos_def();
  }
  
  class ConditionallyIndependentKalmanFilterTest : public ::testing::Test {
   protected:
    ConditionallyIndependentKalmanFilterTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  // This test checks 3 thing:
  // 1) the binomial inverse theorem (BIT) used to compute Finv * error
  // 2) the matrix determinant lemma used to compute log(det(Finv))
  // 3) the computation of the kalman gain from the BIT.
  TEST_F(ConditionallyIndependentKalmanFilterTest, CheckTheMath) {
    // The notation used here follows Durbin and Koopman.
    int ydim = 4;
    int state_dim = 2;
    
    // Residual variance matrix for observed data given state.
    SpdMatrix H(ydim, 0.0);
    H.diag() = pow(rnorm_vector(ydim, 0, 1), 2);
    SpdMatrix Hinv(ydim, 0.0);
    Hinv.diag() = 1.0 / H.diag();

    // State variance matrix.
    SpdMatrix P(state_dim);
    P.randomize();

    // Observation coefficients.
    Matrix Z(ydim, state_dim);  
    Z.randomize();

    // F is the forecast variance: F = H + ZPZ'
    SpdMatrix F_direct = H + Z * P * Z.transpose();
    SpdMatrix Finv_direct = F_direct.inv();

    // An identity matrix.
    SpdMatrix I(state_dim, 1.0);
    Matrix inner = I + P * Z.transpose() * Hinv * Z;

    // Finv as computed by the binomial inverse theorem.
    SpdMatrix Finv = Hinv - Hinv * Z * inner.inv() * P * Z.transpose() * Hinv;

    // This is the check we've been building towards.
    EXPECT_TRUE(MatrixEquals(Finv, Finv_direct));
    
    //-------------------------------------------------------------------------
    // Now verify the "Matrix determinant lemma", used to compute log det(Finv).
    // Check the intermediate calculations used to implement the binomial
    // inverse theorem.
    DenseMatrix Zsparse(Z);
    EXPECT_TRUE(MatrixEquals(Z.transpose() * Hinv * Z,
                             Zsparse.inner(Hinv.diag())));

    LU inner_lu(inner);
    Matrix inner_inv_P = inner_lu.solve(P);
    Vector prediction_error(ydim);
    prediction_error.randomize();
    Vector scaled_error = Hinv * prediction_error
        - Hinv * (Z * inner_inv_P * Z.Tmult(Hinv * prediction_error));

    Vector scaled_error_direct = Finv_direct * prediction_error;
    EXPECT_TRUE(VectorEquals(scaled_error_direct, scaled_error));

    // Make sure LU computes logdet correctly.  This is also tested in the unit
    // test for the LU decomposition.
    EXPECT_NEAR(inner_lu.logdet(), inner.logdet(), 1e-8);

    // Make sure DiagonalMatrix computeds logdet correctly.
    EXPECT_NEAR(Hinv.logdet(), sum(log(Hinv.diag())), 1e-8);
    
    // Check the log determinant of Finv.  This completes the check of the
    // matrix determinant lemma.
    EXPECT_NEAR(
        -1 * inner_lu.logdet() + Hinv.logdet(),
        Finv.logdet(),
        1e-8);

    //-------------------------------------------------------------------------
    // Next, check the calculations for the Kalman gain.
    Matrix transition(state_dim, state_dim);
    transition.randomize();
    Matrix kalman_gain_direct = transition * P * Z.Tmult(Finv);

    Matrix ZtHinv = Z.Tmult(Hinv);
    Matrix kalman_gain = transition * P * (ZtHinv - ZtHinv * Z * inner_inv_P * ZtHinv);
    EXPECT_TRUE(MatrixEquals(kalman_gain, kalman_gain_direct))
        << "kalman_gain_direct: "  << endl
        << kalman_gain_direct
        << "fancy kalman_gain: " << endl
        << kalman_gain;
  }

  // Check that the high- and low-dimensional updates match.
  TEST_F(ConditionallyIndependentKalmanFilterTest, HighLowMatch) {
    int ydim = 6;
    int sample_size = 100;
    int nfactors = 2;
    Matrix data(sample_size, ydim);
    data.randomize();

    NEW(MultivariateStateSpaceModel, model)(ydim);
    for (int i = 0; i < sample_size; ++i) {
      model->add_data(new PartiallyObservedVectorData(data.row(i)));
    }

    NEW(SharedLocalLevelStateModel, state_model)(nfactors, model.get(), ydim);
    state_model->set_initial_state_mean(Vector(nfactors, 0.0));
    state_model->set_initial_state_variance(SpdMatrix(nfactors, 1.0));
    Matrix Beta = state_model->coefficient_model()->Beta();
    Beta.randomize();
    state_model->coefficient_model()->set_Beta(Beta);
    state_model->innovation_model(0)->set_sigsq(20.1);
    state_model->innovation_model(1)->set_sigsq(1.8);
    
    model->add_state(state_model);
    Vector sigma_obs(ydim);
    sigma_obs.randomize();
    model->observation_model()->set_sigsq(sigma_obs * sigma_obs);

    SpdMatrix state_variance(nfactors);
    state_variance.randomize();
    Vector state_mean(nfactors);
    state_mean.randomize();

    Selector observed(ydim, true);
    
    // marg0_lo will use the low_dimensional update.
    Kalman::ConditionallyIndependentMarginalDistribution marg0_lo(
        model.get(), nullptr, 0);
    marg0_lo.set_high_dimensional_threshold_factor(1000);
    marg0_lo.set_state_mean(state_mean);
    marg0_lo.set_state_variance(state_variance);
    marg0_lo.update(data.row(0), observed);

    // marg0_hi will use the high_dimensional update.
    Kalman::ConditionallyIndependentMarginalDistribution marg0_hi(
        model.get(), nullptr, 0);
    marg0_hi.set_high_dimensional_threshold_factor(.01);
    marg0_hi.set_state_mean(state_mean);
    marg0_hi.set_state_variance(state_variance);
    marg0_hi.update(data.row(0), observed);

    EXPECT_TRUE(VectorEquals(marg0_hi.prediction_error(),
                             marg0_lo.prediction_error()));
    EXPECT_TRUE(VectorEquals(marg0_hi.scaled_prediction_error(),
                             marg0_lo.scaled_prediction_error(),
                             1e-4))
        << "low dimensional scaled prediction error at time 0: " << endl
        << marg0_lo.scaled_prediction_error() << endl
        << "high dimensional scaled_prediction_error at time 0: " << endl
        << marg0_hi.scaled_prediction_error() << endl;
        
    EXPECT_NEAR(marg0_hi.forecast_precision_log_determinant(),
                marg0_lo.forecast_precision_log_determinant(),
                1e-4);
    EXPECT_TRUE(MatrixEquals(marg0_hi.kalman_gain(),
                             marg0_lo.kalman_gain()));

    EXPECT_TRUE(is_pos_def(marg0_lo.state_variance()));
    EXPECT_TRUE(is_pos_def(marg0_hi.state_variance()));

    //--------------------------------------------------------------------------
    // 
    
    Kalman::ConditionallyIndependentMarginalDistribution marg1_lo(
        model.get(), &marg0_lo, 1);
    marg1_lo.set_high_dimensional_threshold_factor(1000);
    marg1_lo.set_state_mean(marg0_lo.state_mean());
    marg1_lo.set_state_variance(marg0_lo.state_variance());
    marg1_lo.update(data.row(1), observed);
    
    Kalman::ConditionallyIndependentMarginalDistribution marg1_hi(
        model.get(), &marg0_hi, 1);
    marg1_hi.set_high_dimensional_threshold_factor(1000);
    marg1_hi.set_high_dimensional_threshold_factor(1000);
    marg1_hi.set_state_mean(marg0_hi.state_mean());
    marg1_hi.set_state_variance(marg0_hi.state_variance());
    marg1_hi.update(data.row(1), observed);

    EXPECT_TRUE(VectorEquals(marg1_hi.prediction_error(),
                             marg1_lo.prediction_error()));
    EXPECT_TRUE(VectorEquals(marg1_hi.scaled_prediction_error(),
                             marg1_lo.scaled_prediction_error()));
    EXPECT_NEAR(marg1_hi.forecast_precision_log_determinant(),
                marg1_lo.forecast_precision_log_determinant(),
                1e-7);
    EXPECT_TRUE(MatrixEquals(marg1_hi.kalman_gain(),
                             marg1_lo.kalman_gain()));

    //--------------------------------------------------------------------------
    // Now try again with one missing observation.
    observed.drop(1);
    marg0_lo.update(data.row(0), observed);
    marg0_hi.update(data.row(0), observed);
    EXPECT_TRUE(VectorEquals(marg0_hi.prediction_error(),
                             marg0_lo.prediction_error()));
    EXPECT_TRUE(VectorEquals(marg0_hi.scaled_prediction_error(),
                             marg0_lo.scaled_prediction_error()));
    EXPECT_NEAR(marg0_hi.forecast_precision_log_determinant(),
                marg0_lo.forecast_precision_log_determinant(),
                1e-7);
    EXPECT_TRUE(MatrixEquals(marg0_hi.kalman_gain(),
                             marg0_lo.kalman_gain()));
    
  }

}  // namespace

#include "gtest/gtest.h"
#include "Models/TimeSeries/ArmaModel.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
  bool VectorEquals(const Vector &lhs, const Vector &rhs, double tol = 1e-7) {
    Vector diff = lhs - rhs;
    return diff.max_abs() < tol;
  }

  bool MatrixEquals(const Matrix &lhs, const Matrix &rhs, double tol = 1e-7) {
    Matrix diff = lhs - rhs;
    return diff.max_abs() < tol;
  }

  class ArmaModelTest : public ::testing::Test {
   protected:
    ArmaModelTest() : phi_({.80, .067}), theta_({.53, .09}) {
      GlobalRng::rng.seed(8675309);
    }

    // AR coefficients.
    Vector phi_;
    // MA coefficients.
    Vector theta_;
  };

  TEST_F(ArmaModelTest, Constructor) {
    EXPECT_THROW(ArmaModel(0, 0), std::runtime_error);

    ArmaModel pure_ar(1, 0);
    EXPECT_EQ(1, pure_ar.ar_dimension());
    EXPECT_EQ(0, pure_ar.ma_dimension());

    ArmaModel pure_ma(0, 3);
    EXPECT_EQ(0, pure_ma.ar_dimension());
    EXPECT_EQ(3, pure_ma.ma_dimension());
    
    ArmaModel model(3, 2);
    EXPECT_EQ(3, model.ar_dimension());
    EXPECT_EQ(2, model.ma_dimension());
  }

  TEST_F(ArmaModelTest, Parameters) {
    ArmaModel model(new GlmCoefs(phi_),
                    new VectorParams(theta_),
                    new UnivParams(18));
    EXPECT_TRUE(VectorEquals(phi_, model. ar_coefficients()));
    EXPECT_TRUE(VectorEquals(theta_, model. ma_coefficients()));
    EXPECT_DOUBLE_EQ(18, model.sigsq());
  }

  TEST_F(ArmaModelTest, TransitionMatrix) {
    Vector expanded_phi = concat(phi_, Vector({0, 0}));
    NEW(ArmaStateSpaceTransitionMatrix, T)(expanded_phi);
    EXPECT_EQ(T->nrow(), phi_.size() + 2);
    EXPECT_EQ(T->nrow(), T->ncol());

    Matrix Tdense = T->dense();
    EXPECT_TRUE(VectorEquals(expanded_phi, Tdense.col(0)));

    Matrix Tmanual(4, 4, 0.0);
    Tmanual(0, 0) = phi_[0];
    Tmanual(1, 0) = phi_[1];
    Tmanual(0, 1) = 1.0;
    Tmanual(1, 2) = 1.0;
    Tmanual(2, 3) = 1.0;

    EXPECT_TRUE(MatrixEquals(Tmanual, Tdense))
        << "Tmanual = " << endl << Tmanual
        << "Tdense  = " << endl << Tdense;

    BlockDiagonalMatrix Tblock;
    Tblock.add_block(T);
    
    Vector x(4);
    x.randomize();
    EXPECT_TRUE(VectorEquals(Tblock * x, Tdense * x));
    EXPECT_TRUE(VectorEquals(Tblock.Tmult(x), Tdense.Tmult(x)));
    Vector z1 = x;
    T->multiply_inplace(VectorView(z1));
    EXPECT_TRUE(VectorEquals(z1, Tdense * x));
  }
  
  TEST_F(ArmaModelTest, Stationary) {
    ArmaModel model(new GlmCoefs(phi_),
                    new VectorParams(theta_),
                    new UnivParams(18));
    EXPECT_TRUE(model.is_causal());
    EXPECT_TRUE(model.is_invertible());
    EXPECT_TRUE(model.is_stationary());
  }

  TEST_F(ArmaModelTest, Simulation) {
    ArmaModel model(new GlmCoefs(phi_),
                    new VectorParams(theta_),
                    new UnivParams(18));

    int nobs = 100;
    Vector y = model.simulate(nobs, GlobalRng::rng);
    EXPECT_EQ(nobs, y.size());
    // The residual sd is 18, so the series bounded by a pretty big number.
    EXPECT_LT(y.max_abs(), 60) << "y = " << y;
  }

  TEST_F(ArmaModelTest, LogLikelihood) {
    ArmaModel model(new GlmCoefs(phi_),
                    new VectorParams(theta_),
                    new UnivParams(1.8));
    Vector y = model.simulate(1000, GlobalRng::rng);
    for (int i = 0; i < y.size(); ++i) {
      model.add_data(new DoubleData(y[i]));
    }

    double true_log_likelihood = model.log_likelihood(phi_, theta_, 1.8);
    double wrong_log_likelihood = model.log_likelihood(theta_, phi_, 1.8);
    EXPECT_LT(wrong_log_likelihood, true_log_likelihood);
  }
  
}  // namespace


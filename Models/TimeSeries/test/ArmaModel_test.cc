#include "gtest/gtest.h"
#include "Models/TimeSeries/ArmaModel.hpp"
#include "Models/TimeSeries/ArmaPriors.hpp"
#include "Models/TimeSeries/PosteriorSamplers/ArmaSliceSampler.hpp"
#include "Models/ChisqModel.hpp"
#include "test_utils/test_utils.hpp"

namespace {
  using namespace BOOM;
  using std::endl;
  
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

  TEST_F(ArmaModelTest, Acf) {
    ArmaModel model(new GlmCoefs(phi_),
                    new VectorParams(theta_),
                    new UnivParams(1.0));
    Vector acf = model.acf(10);
    EXPECT_EQ(11, acf.size());
    Vector acf_from_R = {
      1.000000000000000, 0.933447729803580, 0.823580572440072, 0.721405455848898,
      0.632304263032603, 0.554177575967959, 0.485706446397551, 0.425695054707894,
      0.373098375674952, 0.327000269205390, 0.286597806534534};
    EXPECT_TRUE(VectorEquals(acf, acf_from_R))
        << endl
        << "C++ acf: " << acf << endl
        << "R acf  : " << acf_from_R<< endl; 

    Vector autocovariance = model.autocovariance(10);
    EXPECT_EQ(11, autocovariance.size());
    EXPECT_DOUBLE_EQ(autocovariance[3] / autocovariance[2],
                     acf[3] / acf[2]);
    EXPECT_DOUBLE_EQ(autocovariance[9] / autocovariance[7],
                     acf[9] / acf[7]);
  }

  TEST_F(ArmaModelTest, MCMC) {
    double true_sigsq = 2.7;
    ArmaModel model(new GlmCoefs(phi_),
                    new VectorParams(theta_),
                    new UnivParams(true_sigsq));
    Vector y = model.simulate(1000, GlobalRng::rng);

    for (int i = 0; i < y.size(); ++i) {
      model.add_data(new DoubleData(y[i]));
    }

    NEW(UniformArPrior, ar_prior)(phi_.size());
    NEW(UniformMaPrior, ma_prior)(theta_.size());
    NEW(ChisqModel, precision_prior)(3, 2.7);
    NEW(ArmaSliceSampler, sampler)(&model, ar_prior, ma_prior, precision_prior);
    model.set_method(sampler);

    int niter = 100;
    Matrix ar_draws(niter, model.ar_dimension());
    Matrix ma_draws(niter, model.ar_dimension());
     Vector sigma_draws(niter);
     for (int i = 0; i < niter; ++i) {
      //      if (i % 10 == 0) cout << "iteration " << i << endl;
      model.sample_posterior();
      ar_draws.row(i) = model.ar_coefficients();
      ma_draws.row(i) = model.ma_coefficients();
      sigma_draws[i] = model.sigma();
    }

    // TODO: Check the results of these draws with CheckMcmcMatrix once it gets
    // released.

    // std::ofstream ar_out("ar.draws");
    // ar_out << phi_ << endl << ar_draws;

    // std::ofstream ma_out("ma.draws");
    // ma_out << theta_ << endl << ma_draws;

    // std::ofstream sigma_out("sigma.draws");
    // sigma_out << sqrt(true_sigsq) << endl << sigma_draws;
  }
  
}  // namespace


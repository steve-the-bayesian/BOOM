#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/MultivariateRegression.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSampler.hpp"
#include "Models/Glm/PosteriorSamplers/MultivariateRegressionSpikeSlabSampler.hpp"
#include "LinAlg/Cholesky.hpp"

#include "distributions.hpp"
#include "stats/moments.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultivariateRegressionTest : public ::testing::Test {
   protected:
    MultivariateRegressionTest() {
      GlobalRng::rng.seed(8675309);
    }

    void PopulateModel(MultivariateRegressionModel &model,
                       double inclusion_prob = 1.0,
                       int sample_size = 1000) {
      coefficients_ = model.Beta();
      int xdim = coefficients_.nrow();
      int ydim = coefficients_.ncol();
      coefficients_.randomize();
      predictors_.resize(sample_size, xdim);
      response_.resize(sample_size, ydim);

      if (inclusion_prob < 1.0) {
        for (int i = 0; i < xdim; ++i) {
          for (int j = 0; j < ydim; ++j) {
            if (runif() > inclusion_prob) {
              coefficients_(i, j) = 0.0;
            }
          }
        }
      }
      model.set_Beta(coefficients_);
      Sigma_ = model.Sigma();
      Sigma_.randomize();
      model.set_Sigma(Sigma_);

      for (int i = 0; i < sample_size; ++i) {
        Vector predictors(xdim);
        predictors.randomize();
        Vector yhat = predictors * coefficients_;
        Vector response = rmvn(yhat, Sigma_);
        NEW(MvRegData, data_point)(response, predictors);
        model.add_data(data_point);

        predictors_.row(i) = predictors;
        response_.row(i) = response;
      }
    }

    //---------------------------------------------------------------------------
    void SetTrueParameterValues(MultivariateRegressionModel &model) {
      model.set_Beta(coefficients_);
      model.set_Sigma(Sigma_);
    }

    //---------------------------------------------------------------------------
    Ptr<MultivariateRegressionSpikeSlabSampler>
    SetupSpikeSlab(MultivariateRegressionModel &model) {
      int xdim = model.xdim();
      int ydim = model.ydim();
      NEW(WishartModel, residual_precision_prior)(ydim);
      Matrix prior_inclusion_probabilities(xdim, ydim, .25);
      NEW(MatrixVariableSelectionPrior, spike)(prior_inclusion_probabilities);
      NEW(MatrixNormalModel, slab)(xdim, ydim);
      slab->set_mean(Matrix(xdim, ydim, 0.0));
      slab->set_row_variance(SpdMatrix(xdim, 1.0));
      slab->set_column_variance(SpdMatrix(ydim, 1.0));
      NEW(MultivariateRegressionSpikeSlabSampler, sampler)(
          &model, spike, slab, residual_precision_prior);
      model.set_method(sampler);

      SelectorMatrix included_coefficients = model.included_coefficients();
      for (int i = 0; i < coefficients_.nrow(); ++i) {
        for (int j = 0; j < coefficients_.ncol(); ++j) {
          if (fabs(coefficients_(i, j)) < 1e-8) {
            included_coefficients.drop(i, j);
          } else {
            included_coefficients.add(i, j);
          }
        }
      }
      model.Beta_prm()->set_inclusion_pattern(included_coefficients);

      return sampler;
    }

    Matrix coefficients_;
    SpdMatrix Sigma_;
    Matrix predictors_;
    Matrix response_;
  };

  //===========================================================================
  TEST_F(MultivariateRegressionTest, SufficientStatistics) {
    int xdim = 4;
    int ydim = 3;
    int sample_size = 20;
    MultivariateRegressionModel model(xdim, ydim);
    PopulateModel(model, 1.0, sample_size);

    SpdMatrix xtx = predictors_.transpose() * predictors_;
    EXPECT_TRUE(MatrixEquals(xtx, model.suf()->xtx()));

    Matrix xty = predictors_.transpose() * response_;
    EXPECT_TRUE(MatrixEquals(xty, model.suf()->xty()));

    EXPECT_TRUE(VectorEquals(model.suf()->xty().col(1),
                             predictors_.Tmult(response_.col(1))));

    Matrix yty = response_.transpose() * response_;
    EXPECT_TRUE(MatrixEquals(yty, model.suf()->yty()));

    EXPECT_DOUBLE_EQ(predictors_.nrow(), model.suf()->n());

    // Now test 'update raw data'
    Ptr<MvRegSuf> suf = model.suf();
    suf->clear();
    for (int i = 0; i < sample_size; ++i) {
      suf->update_raw_data(response_.row(i), predictors_.row(i), 1.0);
    }
    EXPECT_TRUE(MatrixEquals(xtx, suf->xtx()));
    EXPECT_TRUE(MatrixEquals(xty, suf->xty()));
    EXPECT_TRUE(MatrixEquals(yty, suf->yty()));
    EXPECT_DOUBLE_EQ(predictors_.nrow(), suf->n());
  }

  //===========================================================================
  TEST_F(MultivariateRegressionTest, LogLikelihood) {
    int xdim = 2;
    int ydim = 3;
    int sample_size = 10;
    MultivariateRegressionModel model(xdim, ydim);
    PopulateModel(model, 1.0, sample_size);

    SpdMatrix Siginv = model.Siginv();
    double ldsi = model.ldsi();
    SpdMatrix SSE(ydim, 0.0);

    Matrix predictors(sample_size, xdim);
    double loglike_direct = 0;
    double qform = 0;
    for (int i = 0; i < sample_size; ++i) {
      const Vector &response(model.dat()[i]->y());
      const Vector &predictors(model.dat()[i]->x());
      Vector yhat = model.predict(predictors);
      loglike_direct += dmvn(response, yhat, Siginv, ldsi, true);
      SSE.add_outer(response - yhat);
      qform += Siginv.Mdist(response, yhat);
    }

    EXPECT_TRUE(MatrixEquals(SSE, model.suf()->SSE(model.Beta())));
    EXPECT_NEAR(qform, trace(SSE * Siginv), 1e-5);

    double loglike = model.log_likelihood(model.Beta(), model.Sigma());
    double loglike_inv = model.log_likelihood_ivar(model.Beta(), Siginv);
    EXPECT_NEAR(loglike, loglike_inv, 1e-6);
    EXPECT_NEAR(loglike, loglike_direct, 1e-6);
  }

  //===========================================================================
  TEST_F(MultivariateRegressionTest, McmcConjugatePrior) {
    int xdim = 2;
    int ydim = 3;
    MultivariateRegressionModel model(xdim, ydim);
    int sample_size = 1000;
    PopulateModel(model, 1.0, sample_size);
    NEW(MultivariateRegressionSampler, sampler)(
        &model,
        model.Beta(),
        1.0,
        1.0,
        model.Sigma());
    model.set_method(sampler);

    int niter = 1e+4;
    Matrix beta_draws(niter, xdim * ydim);
    Matrix sigma_draws(niter, ydim * (ydim + 1) / 2);
    Vector true_beta = vec(model.Beta());
    Vector true_sigma = model.Sigma().vectorize();
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      beta_draws.row(i) = vec(model.Beta());
      sigma_draws.row(i) = model.Sigma().vectorize();
    }
    auto status = CheckMcmcMatrix(beta_draws, true_beta);
    EXPECT_TRUE(status.ok) << status.error_message();
    status = CheckMcmcMatrix(sigma_draws, true_sigma);
    EXPECT_TRUE(status.ok) << status.error_message();
  }

  //===========================================================================
  // Check that the Cholesky decomposition of siginv \otimes ominv is computed
  // correctly by CompositeCholesky, as compared to brute force.
  TEST_F(MultivariateRegressionTest, CompositeCholeskyTest) {
    SpdMatrix siginv(3);
    siginv.randomize();
    SpdMatrix ominv(5);
    ominv.randomize();

    SpdMatrix full_matrix = Kronecker(siginv, ominv);
    Cholesky full_cholesky(full_matrix);

    //----------------------------------------------------------------------
    // First check that the computation is correct when all variables are
    // included.
    SelectorMatrix included(ominv.nrow(), siginv.nrow(), true);
    CompositeCholesky composite_cholesky(ominv.chol(), siginv.chol(), included);

    EXPECT_TRUE(MatrixEquals(full_cholesky.getL(),
                             Kronecker(siginv.chol(), ominv.chol())))
        << "Numerical problems!";

    EXPECT_TRUE(MatrixEquals(full_cholesky.getL(),
                             composite_cholesky.matrix()))
        << "Full chol: " << endl
        << full_cholesky.getL() << endl
        << "Composite: " << endl
        << composite_cholesky.matrix()
        << "Raw Kronecker product: " << endl
        << Kronecker(siginv.chol(), ominv.chol());

    Vector x(full_cholesky.nrow());
    x.randomize();

    Vector y1 = full_matrix.solve(x);
    Vector y2 = composite_cholesky.solve(x);
    EXPECT_TRUE(VectorEquals(y1, y2, 1e-4))
        << endl
        << "right answer: " << y1 << endl
        << "answer given: " << y2 << endl
        << "max |right - answer| = " << (y1 - y2).max_abs() << endl;

    EXPECT_NEAR(composite_cholesky.Mdist(y2),
                full_matrix.Mdist(y2),
                1e-4);

    //----------------------------------------------------------------------
    // Now exclude some variables and check again.
    included.randomize();
    Selector inc(included.vectorize());

    SpdMatrix full_subset(inc.select(full_matrix));
    Matrix full_subset_cholesky = full_subset.chol();

    CompositeCholesky composite_subset_cholesky(
        ominv.chol(), siginv.chol(), included);
    EXPECT_TRUE(MatrixEquals(full_subset_cholesky,
                             composite_subset_cholesky.matrix()))
        << "Full subset cholesky: " << endl
        << full_subset_cholesky << endl
        << "Composite subset cholesky: " << endl
        << composite_subset_cholesky.matrix()
        << "full_cholesky * full_cholesky.transpose: " << endl
         << full_subset_cholesky.outer() << endl
        << "composite_cholesky.outer() " << endl
        << composite_cholesky.matrix().outer() << endl
        << "difference: " << endl
        << full_subset_cholesky.outer()
        - composite_subset_cholesky.matrix().outer();

    EXPECT_TRUE(MatrixEquals(composite_subset_cholesky.matrix().outer(),
                             full_subset));
    Vector v(full_subset.nrow());
    v.randomize();
    EXPECT_NEAR(composite_subset_cholesky.Mdist(v),
                full_subset.Mdist(v),
                1e-8);
  }

  //===========================================================================
  TEST_F(MultivariateRegressionTest, ThingsAreRightSize) {
    int xdim = 12;
    int ydim = 3;
    MultivariateRegressionModel model(xdim, ydim);
    EXPECT_EQ(xdim, model.xdim());
    EXPECT_EQ(ydim, model.ydim());
    EXPECT_EQ(xdim, model.Beta().nrow());
    EXPECT_EQ(ydim, model.Beta().ncol());
    EXPECT_EQ(xdim, model.included_coefficients().nrow());
    EXPECT_EQ(ydim, model.included_coefficients().ncol());

    EXPECT_EQ(ydim, model.Sigma().nrow());
    EXPECT_EQ(ydim, model.Sigma().ncol());

    EXPECT_EQ(ydim, model.Siginv().nrow());
    EXPECT_EQ(ydim, model.Siginv().ncol());

    EXPECT_EQ(ydim, nrow(model.residual_precision_cholesky()));
  }


  //===========================================================================
  TEST_F(MultivariateRegressionTest, SpikeSlabDrawSigmaTest) {
    int xdim = 12;
    int ydim = 3;
    MultivariateRegressionModel model(xdim, ydim);
    int sample_size = 1000;
    PopulateModel(model, .25, sample_size);
    Ptr<MultivariateRegressionSpikeSlabSampler> sampler = SetupSpikeSlab(model);
    int niter = 1000;

    Matrix beta_draws(niter, xdim * ydim);
    Matrix sigma_draws(niter, ydim * (ydim + 1) / 2);
    for (int i = 0; i < niter; ++i) {
      sampler->draw_residual_variance();
      // beta draws should be constant
      beta_draws.row(i) = vec(model.Beta());
      sigma_draws.row(i) = model.Sigma().vectorize();
    }

    EXPECT_TRUE(MatrixEquals(var(beta_draws), SpdMatrix(beta_draws.ncol(), 0.0)));
    EXPECT_TRUE(VectorEquals(beta_draws.row(0), vec(coefficients_)));

    auto status = CheckMcmcMatrix(sigma_draws, Sigma_.vectorize());
    EXPECT_TRUE(status.ok) << status.error_message();
  }
  //===========================================================================
  TEST_F(MultivariateRegressionTest, SpikeSlabDrawBetaTest) {
    int xdim = 12;
    int ydim = 3;
    MultivariateRegressionModel model(xdim, ydim);
    EXPECT_EQ(ydim, model.residual_precision_cholesky().nrow());
    EXPECT_EQ(ydim, model.residual_precision_cholesky().ncol());

    int sample_size = 1000;
    PopulateModel(model, .25, sample_size);
    Ptr<MultivariateRegressionSpikeSlabSampler> sampler = SetupSpikeSlab(model);
    int niter = 1000;

    Matrix beta_draws(niter, xdim * ydim);
    Matrix sigma_draws(niter, ydim * (ydim + 1) / 2);
    sampler->set_total_row_precision_cholesky();
    for (int i = 0; i < niter; ++i) {
      sampler->draw_coefficients();
      beta_draws.row(i) = vec(model.Beta());
      sigma_draws.row(i) = model.Sigma().vectorize();
    }

    EXPECT_TRUE(MatrixEquals(var(sigma_draws), SpdMatrix(sigma_draws.ncol(), 0.0)));
    EXPECT_TRUE(VectorEquals(sigma_draws.row(0), Sigma_.vectorize()));

    auto status = CheckMcmcMatrix(beta_draws, vec(coefficients_));
    EXPECT_TRUE(status.ok) << status.error_message();
  }

  //===========================================================================
  TEST_F(MultivariateRegressionTest, VariableSelectorTest) {
    int xdim = 12;
    int ydim = 3;
    MultivariateRegressionModel model(xdim, ydim);
    EXPECT_EQ(ydim, model.residual_precision_cholesky().nrow());
    EXPECT_EQ(ydim, model.residual_precision_cholesky().ncol());

    int sample_size = 1000;
    Sigma_ *= 1e-8;
    PopulateModel(model, .25, sample_size);
    Ptr<MultivariateRegressionSpikeSlabSampler> sampler = SetupSpikeSlab(model);
    int niter = 1000;
    int burn = 100;

    Matrix inclusion_draws(niter, xdim * ydim);
    SetTrueParameterValues(model);
    sampler->set_total_row_precision_cholesky();
    for (int i = 0; i < burn; ++i) {
      sampler->draw_inclusion_indicators();
    }
    for (int i = 0; i < niter; ++i) {
      sampler->draw_inclusion_indicators();
      inclusion_draws.row(i) =
          model.included_coefficients().vectorize().to_Vector();
    }
    Vector inclusion_probs = mean(inclusion_draws);
    SelectorMatrix true_inclusion_indicators(coefficients_.nrow(),
                                             coefficients_.ncol());
    for (int i = 0; i < coefficients_.nrow(); ++i) {
      for (int j = 0; j < coefficients_.ncol(); ++j) {
        bool in = fabs(coefficients_(i, j)) > 1e-6;
        if (in) {
          true_inclusion_indicators.add(i, j);
        } else {
          true_inclusion_indicators.drop(i, j);
        }
      }
    }
    Selector true_inclusion_vector = true_inclusion_indicators.vectorize();
    double success_count = 0.0;
    for (int i = 0; i < inclusion_probs.size(); ++i) {
      if (true_inclusion_vector[i]) {
        success_count += inclusion_probs[i] > .5;
      } else {
        success_count += inclusion_probs[i] < .5;
      }
    }
    double success_rate = success_count / inclusion_probs.size();
    EXPECT_GT(success_rate, .9)
        << std::endl
        << "Fraction of correctly identified variables is below .9."
        << std::endl
        << cbind(true_inclusion_vector.to_Vector(),
                 inclusion_probs);
  }

  //===========================================================================
  TEST_F(MultivariateRegressionTest, SpikeSlabTest) {
    int xdim = 12;
    int ydim = 3;
    MultivariateRegressionModel model(xdim, ydim);
    int sample_size = 1000;
    PopulateModel(model, .25, sample_size);
    SetupSpikeSlab(model);

    int niter = 1000;
    Matrix beta_draws(niter, xdim * ydim);
    Matrix sigma_draws(niter, ydim * (ydim + 1) / 2);
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      beta_draws.row(i) = vec(model.Beta());
      sigma_draws.row(i) = model.Sigma().vectorize();
    }
    auto status = CheckMcmcMatrix(beta_draws, vec(coefficients_), .95,
                                  true, "beta.draws");
    EXPECT_TRUE(status.ok) << status.error_message();
    status = CheckMcmcMatrix(sigma_draws, Sigma_.vectorize(true), .95,
                             true, "sigma.draws");
    EXPECT_TRUE(status.ok) << status.error_message();
  }

}  // namespace

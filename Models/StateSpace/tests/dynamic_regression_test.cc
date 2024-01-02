#include "gtest/gtest.h"

#include "Models/StateSpace/DynamicRegression.hpp"
#include "Models/StateSpace/PosteriorSamplers/DynamicRegressionDirectGibbs.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include "Models/StateSpace/tests/state_space_test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"

#include <fstream>

namespace {
  using namespace BOOM;
  using namespace BOOM::StateSpace;
  using std::endl;
  using std::cout;

  RegressionDataTimePoint simulate_data(
      int sample_size,
      const Vector &beta,
      double residual_sd) {
    RegressionDataTimePoint ans;
    for (int i = 0; i < sample_size; ++i) {
      Vector x(beta.size());
      x.randomize();
      x[0] = 1.0;
      double y = rnorm(beta.dot(x), residual_sd);
      NEW(RegressionData, dp)(y, x);
      ans.add_data(dp);
    }
    return ans;
  }

  //===========================================================================
  class RegressionDataTimePointTest : public ::testing::Test {
   protected:
    RegressionDataTimePointTest() {}
  };

  TEST_F(RegressionDataTimePointTest, ConstructorTest) {
    NEW(RegressionDataTimePoint, dp)(3);
    EXPECT_EQ(3, dp->xdim());
  }

  TEST_F(RegressionDataTimePointTest, AddDataTest) {
    NEW(RegressionDataTimePoint, data)(3);
    Selector omit_one("101");
    Selector keep_all("111");
    Selector keep_none("000");

    auto suf = data->xtx_xty(keep_all);
    EXPECT_EQ(suf.first.nrow(), 3);

    NEW(RegressionData, dp)(1.8, rnorm_vector(3, 0, 1));
    data->add_data(dp);
    EXPECT_EQ(1, data->sample_size());
    suf = data->xtx_xty(keep_all);
    EXPECT_TRUE(MatrixEquals(suf.first, dp->x().outer()))
        << "\n suf: \n" << suf.first
        << "\n x.outer: \n" << dp->x().outer();
    EXPECT_TRUE(VectorEquals(suf.second, dp->x() * dp->y()));

    NeRegSuf reg_suf(3);
    reg_suf.update(dp);

    suf = data->xtx_xty(omit_one);
    EXPECT_TRUE(MatrixEquals(suf.first, reg_suf.xtx(omit_one)));
    EXPECT_TRUE(VectorEquals(suf.second, reg_suf.xty(omit_one)));

    suf = data->xtx_xty(keep_none);
    EXPECT_TRUE(MatrixEquals(suf.first, reg_suf.xtx(keep_none)));
    EXPECT_TRUE(VectorEquals(suf.second, reg_suf.xty(keep_none)));

    for (int i = 0; i < 20; ++i) {
      NEW(RegressionData, rd)(rnorm(), rnorm_vector(3, 0, 1));
      data->add_data(rd);
      reg_suf.update(rd);

      suf = data->xtx_xty(keep_all);
      EXPECT_DOUBLE_EQ(data->yty(), reg_suf.yty());
      EXPECT_EQ(reg_suf.n(), data->sample_size());
      EXPECT_TRUE(MatrixEquals(suf.first, reg_suf.xtx()));
      EXPECT_TRUE(VectorEquals(suf.second, reg_suf.xty()));

      suf = data->xtx_xty(omit_one);
      EXPECT_TRUE(MatrixEquals(suf.first, reg_suf.xtx(omit_one)));
      EXPECT_TRUE(VectorEquals(suf.second, reg_suf.xty(omit_one)));

      suf = data->xtx_xty(keep_none);
      EXPECT_TRUE(MatrixEquals(suf.first, reg_suf.xtx(keep_none)));
      EXPECT_TRUE(VectorEquals(suf.second, reg_suf.xty(keep_none)));
    }
  }

  //===========================================================================
  class ProductSelectorMatrixTest: public ::testing::Test {
   protected:
    ProductSelectorMatrixTest() {
      inc1_ = Selector("110010");
      inc2_ = Selector("010011");

      SetDense(inc1_, inc2_);
      dense1_ = Matrix(6, 3, 0.0);
      dense1_(0, 0) = 1;
      dense1_(1, 1) = 1;
      dense1_(4, 2) = 1;

      dense2_ = Matrix(6, 3, 0.0);
      dense2_(1, 0) = 1;
      dense2_(4, 1) = 1;
      dense2_(5, 2) = 1;
    }

    void SetDense(const Selector &inc1, const Selector &inc2) {
      SpdMatrix Id(inc1.nvars_possible(), 1.0);
      dense1_ = inc1.select_cols(Id);
      dense2_ = inc2.select_cols(Id);
      dense_ = dense2_.transpose() * dense1_;
    }

    Selector inc1_;
    Selector inc2_;
    Matrix dense1_;
    Matrix dense2_;
    Matrix dense_;
  };

  TEST_F(ProductSelectorMatrixTest, TestDense) {
    ProductSelectorMatrix mat(inc1_, inc2_);
    EXPECT_EQ(dense_.nrow(), mat.nrow());
    EXPECT_EQ(dense_.ncol(), mat.ncol());
    EXPECT_TRUE(MatrixEquals(dense_, mat.dense()))
        << "\nDense = \n" << dense_
        << "\nmat.dense = \n" << mat.dense();
  }

  TEST_F(ProductSelectorMatrixTest, TestProduct) {
    ProductSelectorMatrix mat(inc1_, inc2_);
    Vector y(inc1_.nvars());
    y.randomize();

    EXPECT_TRUE(VectorEquals(mat * y, dense_ * y));
  }

  TEST_F(ProductSelectorMatrixTest, TestSandwich) {
    inc1_ = Selector("111011");
    inc2_ = Selector("101111");
    ProductSelectorMatrix mat(inc1_, inc2_);
    SetDense(inc1_, inc2_);

    int dim = dense_.ncol();
    SpdMatrix Sigma(dim);
    Sigma.randomize();

    SpdMatrix expected = dense_ * Sigma * dense_.transpose();
    SpdMatrix observed = mat.sandwich(Sigma);
    EXPECT_TRUE(MatrixEquals(expected, observed));

    Vector y(inc1_.nvars());
    y.randomize();
    DiagonalMatrix diag(y);
    EXPECT_TRUE(MatrixEquals(
        mat.sandwich(diag),
        dense_ * diag * dense_.transpose()));
  }

  TEST_F(ProductSelectorMatrixTest, TestTranspose) {
    ProductSelectorMatrix mat(inc1_, inc2_);
    EXPECT_TRUE(MatrixEquals(dense_.transpose(), mat.transpose().dense()));
  }
  //===========================================================================

  class DynamicRegressionKalmanFilterNodeTest : public ::testing::Test {
   protected:
    using Node = DynamicRegressionKalmanFilterNode;

    DynamicRegressionKalmanFilterNodeTest() {
      GlobalRng::rng.seed(8675309);

      prior_mean_ = {1.0, 2.0, -7.0};
      prior_variance_ = SpdMatrix(3);
      prior_variance_.randomize();
      prior_precision_ = prior_variance_.inv();
      residual_sd_ = 1.8;
      residual_variance_ = square(residual_sd_);
    }

    Vector prior_mean_;
    SpdMatrix prior_variance_;
    SpdMatrix prior_precision_;
    double residual_sd_;
    double residual_variance_;
  };

  // Basic functionality testing for the 'initialize' method.
  TEST_F(DynamicRegressionKalmanFilterNodeTest, InitializeTest) {
    Node node;
    EXPECT_TRUE(VectorEquals(node.state_mean(), Vector(1, 0.0)));
    EXPECT_TRUE(MatrixEquals(node.unscaled_state_precision(),
                             SpdMatrix(1, 1.0)));

    RegressionDataTimePoint data = simulate_data(
        100, prior_mean_, residual_sd_);

    Selector keep_all("111");
    double marginal_loglike = node.initialize(
        keep_all, prior_mean_, prior_precision_, data, residual_variance_);
    EXPECT_TRUE(std::isfinite(marginal_loglike));

    EXPECT_EQ(3, node.state_mean().size());
    EXPECT_EQ(3, node.unscaled_state_precision().nrow());
    EXPECT_GT(node.unscaled_state_precision().logdet(),
              prior_precision_.logdet());

    Selector drop_middle("101");
    node.initialize(drop_middle, prior_mean_, prior_precision_, data,
                    residual_variance_);
    EXPECT_EQ(2, node.state_mean().size());
    EXPECT_EQ(2, node.unscaled_state_precision().nrow());
  }

  // This is a smoke test to see if the 'update' method runs without crashing.
  TEST_F(DynamicRegressionKalmanFilterNodeTest, UpdateTest) {
    Node now;
    Node prev;

    Selector keep_all("111");
    NEW(RegressionDataTimePoint, data_0)(simulate_data(
        100, prior_mean_, residual_sd_));
    NEW(RegressionDataTimePoint, data_1)(simulate_data(
        100, prior_mean_, residual_sd_));

    prev.initialize(keep_all, prior_mean_, prior_precision_, *data_0,
                    residual_variance_);

    DynamicRegressionModel model(prior_mean_.size());
    model.add_data(data_0);
    model.add_data(data_1);

    now.update(prev, *data_1, model, 1);
  }

  TEST_F(DynamicRegressionKalmanFilterNodeTest, SimulateTest) {
    // test the simulate_coefficients method.
  }

  //===========================================================================
  class DynamicRegressionModelTest : public ::testing::Test {
   protected:
    DynamicRegressionModelTest() {
      GlobalRng::rng.seed(8675309);
    }

  };

  // Check the getters and setters for initial_state_mean and the various
  // variance parameters.
  TEST_F(DynamicRegressionModelTest, InitialStateDistributionTest) {
    DynamicRegressionModel model(4);

    EXPECT_EQ(4, model.initial_state_mean().size());
    EXPECT_EQ(4, model.unscaled_initial_state_precision().nrow());

    Vector mean = {4, 3, 2, 1};
    model.set_initial_state_mean(mean);
    EXPECT_TRUE(VectorEquals(mean, model.initial_state_mean()));

    SpdMatrix variance(4);
    variance.randomize();
    model.set_unscaled_initial_state_variance(variance);
    EXPECT_TRUE(MatrixEquals(
        variance.inv(),
        model.unscaled_initial_state_precision()));

    model.set_residual_variance(2.3);
    EXPECT_DOUBLE_EQ(2.3, model.residual_variance());
    EXPECT_DOUBLE_EQ(sqrt(2.3), model.residual_sd());
    double sd = model.residual_sd();
    EXPECT_DOUBLE_EQ(model.residual_variance(), sd * sd);
  }

  // Checks set_inclusion_indicators and set_included_coefficients.
  TEST_F(DynamicRegressionModelTest, CoefficientTest) {
    DynamicRegressionModel model(3);
    EXPECT_EQ(0, model.time_dimension());
    EXPECT_EQ(3, model.xdim());

    Vector beta(3);
    beta.randomize();
    double sigma = 1.2;
    for (int i = 0; i < 10; ++i) {
      NEW(RegressionDataTimePoint, time_point)(simulate_data(
          1, beta, sigma));
      model.add_data(time_point);
    }
    EXPECT_EQ(10, model.time_dimension());

    EXPECT_EQ(0, model.inclusion_indicators(4).nvars());
    EXPECT_EQ(3, model.inclusion_indicators(4).nvars_possible());
    EXPECT_EQ(0, model.included_coefficients(4).size());

    Selector inc("101");
    model.set_inclusion_indicators(4, inc);
    EXPECT_EQ(2, model.inclusion_indicators(4).nvars());
    EXPECT_EQ(3, model.inclusion_indicators(4).nvars_possible());
    EXPECT_EQ(2, model.included_coefficients(4).size());

    model.set_included_coefficients(4, Vector{1.2, 2.3});
    EXPECT_TRUE(VectorEquals(model.included_coefficients(4),
                             Vector{1.2, 2.3}));
  }

  // Test that the coefficients change after imputation.
  TEST_F(DynamicRegressionModelTest, ImputationTest) {
  }

  //===========================================================================
  class DynamicRegressionDirectGibbsTest : public ::testing::Test {
   protected:
    DynamicRegressionDirectGibbsTest() {
    }

    Selector simulate_inc(const Selector &inc_prev,
                          const std::vector<Matrix> &transition_probabilities) {
      Selector inc = inc_prev;
      for (int i = 0; i < inc.nvars_possible(); ++i) {
        bool included = runif() < transition_probabilities[i](inc_prev[i], 1);
        if (included) {
          inc.add(i);
        } else {
          inc.drop(i);
        }
      }
      return inc;
    }

    Vector simulate_beta(const Vector &beta_prev, const Selector &inc,
                         const Vector &innovation_sd) {
      Vector beta = beta_prev;
      for (int i = 0; i < beta.size(); ++i) {
        if (inc[i]) {
          beta[i] += rnorm(0, innovation_sd[i]);
        } else {
          beta[i] = 0;
        }
      }
      return beta;
    }

  };

  TEST_F(DynamicRegressionDirectGibbsTest, InferMarkovPriorTest) {
    using Sampler = DynamicRegressionDirectGibbsSampler;
    Matrix P = Sampler::infer_Markov_prior(.3, 8, 1.0);
    EXPECT_NEAR(P.row(0).sum(), 1.0, 1e-6);
    EXPECT_NEAR(P.row(1).sum(), 1.0, 1e-6);

    P = Sampler::infer_Markov_prior(.85, 20, 1.0);
    EXPECT_NEAR(P.row(0).sum(), 1.0, 1e-6);
    EXPECT_NEAR(P.row(1).sum(), 1.0, 1e-6);

    P = Sampler::infer_Markov_prior(.6, 12, 1.0);
    EXPECT_NEAR(P.row(0).sum(), 1.0, 1e-6);
    EXPECT_NEAR(P.row(1).sum(), 1.0, 1e-6);
  }

  TEST_F(DynamicRegressionDirectGibbsTest, McmcTest) {
    int time_dimension = 20;
    int sample_size_per_period = 100;
    int xdim = 3;
    double residual_sd = 1.2;

    Vector innovation_sd = {.002, .001, .0004};
    Vector b0 = {3.0, -1.7, 9.8};
    Vector beta_prev = b0;
    Selector g0("111");
    Selector inc_prev = g0;

    Vector stationary_probabilities = {.999, .999, .999};
    Vector expected_durations = {800, 2000, 1200};
    std::vector<Matrix> transition_probabilities;
    for (int i = 0; i < 3; ++i) {
      transition_probabilities.push_back(
          DynamicRegressionDirectGibbsSampler::infer_Markov_prior(
              stationary_probabilities[i],
              expected_durations[i],
              1.0));
      std::cout << "transition_probabilities[" << i << "] = \n"
                << transition_probabilities.back()
                << "\n";
    }

    NEW(DynamicRegressionModel, model)(xdim);
    Matrix beta_path(xdim, time_dimension);

    for (int t = 0; t < time_dimension; ++t) {
      beta_path.col(t) = beta_prev;
      NEW(RegressionDataTimePoint, time_point)(
          simulate_data(sample_size_per_period, beta_prev, residual_sd));
      model->add_data(time_point);
      Selector inc = simulate_inc(inc_prev, transition_probabilities);
      Vector beta = simulate_beta(beta_prev, inc, innovation_sd);
      beta_prev = beta;
      inc_prev = inc;
    }
    std::cout << "True beta path: " << std::endl
              << beta_path.transpose() << std::endl;

    NEW(DynamicRegressionDirectGibbsSampler, sampler)(
        model.get(),
        1.0,
        residual_sd,
        innovation_sd,
        Vector{1, 1, 1},
        stationary_probabilities,
        expected_durations,
        Vector{1, 1, 1});
    model->set_method(sampler);

    int niter = 100;
    Vector residual_sd_draws(niter);
    Matrix innovation_sd_draws(niter, xdim);
    Matrix final_beta_draws(niter, xdim);

    model->set_residual_variance(square(residual_sd));
    for (int j = 0; j < xdim; ++j) {
      model->innovation_error_model(j)->set_sigsq(square(innovation_sd[j]));
      model->transition_model(j)->set_Q(transition_probabilities[j]);
    }

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      residual_sd_draws[i] = model->residual_sd();
      innovation_sd_draws.row(i) = sqrt(model->unscaled_innovation_variances());
      final_beta_draws.row(i) = model->coef(time_dimension - 1).Beta();
    }
    EXPECT_NE(residual_sd_draws[0], residual_sd_draws.back())
        << "residual_sd_draws = "
        << residual_sd_draws;
    EXPECT_NE(innovation_sd_draws(0, 0), innovation_sd_draws(niter - 1, 0))
        << "innovation_sd_draws = "
        << innovation_sd_draws;

    std::cout << final_beta_draws << beta_path.last_col() << std::endl;
  }

}  // namespace

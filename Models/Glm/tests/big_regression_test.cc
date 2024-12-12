#include "gtest/gtest.h"

#include "Models/MvnGivenScalarSigma.hpp"
#include "Models/ChisqModel.hpp"
#include "Models/Glm/VariableSelectionPrior.hpp"
#include "distributions.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "Models/Glm/RegressionSlabPrior.hpp"
#include "Models/Glm/PosteriorSamplers/BregVsSampler.hpp"
#include "Models/Glm/PosteriorSamplers/AdaptiveSpikeSlabRegressionSampler.hpp"
#include "Models/Glm/PosteriorSamplers/BigAssSpikeSlabSampler.hpp"

#include "test_utils/test_utils.hpp"
#include "stats/AsciiDistributionCompare.hpp"
#include "stats/ECDF.hpp"
#include "cpputil/seq.hpp"
#include "cpputil/DateTime.hpp"

#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;

  inline double inclusion_probability(const ConstVectorView &coefficients) {
    double ans = 0;
    for (auto y : coefficients) {
      ans += (y != 0);
    }
    return ans / coefficients.size();
  }

  class BigRegressionTest : public ::testing::Test {
   protected:

    BigRegressionTest()
        : residual_sd_(0.03)
    {
      GlobalRng::rng.seed(8675309);
    }

    void SimulateCoefficients(int num_nonzero) {
      coefficients_.resize(predictors_.ncol());
      coefficients_.randomize();
      if (num_nonzero >= coefficients_.size()) {
        return;
      }
      VectorView zero_coefficients(coefficients_, num_nonzero);
      zero_coefficients = 0.0;
    }

    void SimulatePredictors(int nobs, int xdim) {
      predictors_.resize(nobs, xdim);
      predictors_.randomize();
      predictors_.col(0) = 1.0;
    }

    void SimulateResponse() {
      response_ = predictors_ * coefficients_;
      for (int i = 0; i < response_.size(); ++i) {
        response_[i] += rnorm_mt(GlobalRng::rng, 0, residual_sd_);
      }
    }

    void FillRegressionData() {
      for (int i = 0; i < response_.size(); ++i) {
        regression_data_.push_back(new RegressionData(
            response_[i], predictors_.row(i)));
      }
    }

    Matrix predictors_;
    Vector response_;
    double residual_sd_;
    Vector coefficients_;
    std::vector<Ptr<RegressionData>> regression_data_;
  };

  TEST_F(BigRegressionTest, Chunks) {
    int total_predictor_dim = 10;
    int max_model_dim = 3;
    NEW(BigRegressionModel, model)(total_predictor_dim, max_model_dim);
    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(total_predictor_dim, .1);
    NEW(RegressionSlabPrior, slab)(SpdMatrix(1), model->Sigsq_prm(),
                                   1.0, 1.0, 5.0, 0.1);
    NEW(BigAssSpikeSlabSampler, sampler)(model.get(), spike, slab,
                                         residual_precision_prior);
    model->set_method(sampler);

    EXPECT_EQ(model->subordinate_model(0)->xdim(), 4);
    EXPECT_EQ(model->subordinate_model(1)->xdim(), 4);
    EXPECT_EQ(model->subordinate_model(2)->xdim(), 4);

    Vector x = seq<double>(0.0, 9.0);
    EXPECT_EQ(x.size(), total_predictor_dim);

    EXPECT_TRUE(VectorEquals(sampler->select_chunk(x, 0),
                             Vector{0.0, 1.0, 2.0, 3.0}));
    EXPECT_TRUE(VectorEquals(sampler->select_chunk(x, 1),
                             Vector{4.0, 5.0, 6.0}));
  }

  TEST_F(BigRegressionTest, StreamDataForInitialScreen) {
    int total_predictor_dim = 6;
    int max_model_dim = 3;
    NEW(BigRegressionModel, model)(total_predictor_dim, max_model_dim);
    Vector x(total_predictor_dim);
    x.randomize();
    x[0] = 1.0;
    double y = 3.0;
    NEW(RegressionData, data_point)(y, x);
    model->stream_data_for_initial_screen(*data_point);

    EXPECT_EQ(model->number_of_subordinate_models(), 2);
    RegressionModel *m0 = model->subordinate_model(0);
    RegressionModel *m1 = model->subordinate_model(1);

    EXPECT_EQ(m0->xdim(), 4);
    EXPECT_EQ(m1->xdim(), 3);

    ConstVectorView x0(x, 0, 4);
    EXPECT_TRUE(MatrixEquals(m0->suf()->xtx(), Vector(x0).outer()));
    EXPECT_TRUE(VectorEquals(m0->suf()->xty(), x0 * y));

    Vector x1 = Vector(1, 1.0).concat(ConstVectorView(x, 4));
    EXPECT_TRUE(MatrixEquals(m1->suf()->xtx(), x1.outer()))
        << "m1->suf()->xtx() = \n"
        << m1->suf()->xtx()
        << "\ndirect calculation = \n"
        << x1.outer();
    EXPECT_TRUE(VectorEquals(m1->suf()->xty(), x1 * y));
  }

  // A setting where there are some obvious variables for the sampler to find,
  // with some obviously important variables located in different shards.
  TEST_F(BigRegressionTest, FindsRightVariables) {
    int max_model_dim = 40;
    int total_predictor_dim = 100;
    int sample_size = 1000;
    SimulatePredictors(sample_size, total_predictor_dim);
    coefficients_ = Vector(predictors_.ncol(), 0.0);
    coefficients_[0] = 12;
    coefficients_[3] = -50;
    coefficients_[22] = 90;
    coefficients_[93] = 200;
    SimulateResponse();
    FillRegressionData();

    // std::cout << "Should see significance in positions 0, 3, 22, 93\n";

    NEW(BigRegressionModel, model)(total_predictor_dim, max_model_dim);
    for (int i = 0; i < sample_size; ++i) {
      model->stream_data_for_initial_screen(*regression_data_[i]);
    }

    Vector global_xty = response_ * predictors_;
    RegressionModel *m0 = model->subordinate_model(0);
    RegressionModel *m1 = model->subordinate_model(
        model->number_of_subordinate_models() - 1);

    EXPECT_TRUE(VectorEquals(ConstVectorView(global_xty, 0, m0->xdim()),
                             m0->suf()->xty()));
    EXPECT_NEAR(global_xty.back(),
                m1->suf()->xty().back(),
                1e-8);

    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(total_predictor_dim,
                                       1.0 / total_predictor_dim);
    NEW(RegressionSlabPrior, slab_prototype)(
        SpdMatrix(1.0),
        model->Sigsq_prm(),
        1.0,
        1.0,
        5.0,
        0.1);

    NEW(BigAssSpikeSlabSampler, sampler)(
        model.get(), spike, slab_prototype, residual_precision_prior);
    model->set_method(sampler);

    bool use_threads = true;
    double inclusion_threshold = .10;
    sampler->initial_screen(100, inclusion_threshold, use_threads);
    Selector inc0 = model->subordinate_model(0)->inc();
    EXPECT_TRUE(inc0[0]);
    EXPECT_TRUE(inc0[3]);
    EXPECT_TRUE(inc0[22]);
    for (int i = 1; i < 3; ++i) EXPECT_FALSE(inc0[i]);
    for (int i = 4; i < 22; ++i) EXPECT_FALSE(inc0[i]);
    for (int i = 23; i < inc0.nvars_possible(); ++i) EXPECT_FALSE(inc0[i]);

    Selector inc1 = model->subordinate_model(1)->inc();
    EXPECT_TRUE(inc1[0]);
    for (int i = 1; i < inc1.nvars_possible(); ++i) {
      EXPECT_FALSE(inc1[i]);
    }

    Selector inc2 = model->subordinate_model(2)->inc();
    EXPECT_TRUE(inc2[0]);
    // pos93 is the position of global predictor variable 93 in subordinate_model 2.
    int pos93 = inc2.nvars_possible() - 7;
    EXPECT_TRUE(inc2[pos93]);
    for( int i = 1; i < pos93; ++i) EXPECT_FALSE(inc2[i]);
    for( int i = pos93 + 1; i < inc2.nvars_possible(); ++i) EXPECT_FALSE(inc2[i]);

    Selector good_ones(std::vector<int>{0, 3, 22, 93}, total_predictor_dim);
    SpdMatrix xtx(4, 0.0);
    for (int i = 0; i < regression_data_.size(); ++i) {
      xtx.add_outer(good_ones.select(regression_data_[i]->x()));
      model->stream_data_for_restricted_model(*regression_data_[i]);
    }
    EXPECT_TRUE(MatrixEquals(xtx, model->restricted_model()->suf()->xtx()));

    EXPECT_EQ(good_ones, model->candidate_selector());

    int niter = 100;
    Vector sigma_draws(niter);
    Matrix beta_draws(niter, total_predictor_dim);
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
    }

    for (int el : std::vector<int>{0, 3, 22, 93}) {
      const ConstVectorView beta(beta_draws.col(el));
      EXPECT_GE(inclusion_probability(beta), .90) << SubMatrix(beta_draws, 90, 99, 0, 8);
    }
  }

  //===========================================================================
  // The whole point of the big spike and slab is to handle wide data.  This is
  // a test with some wide data.
  TEST_F(BigRegressionTest, ReallyBig) {
    int max_model_dim = 500;
    int total_predictor_dim = 10000;
    int sample_size = 5000;

    std::cout << "Simulating predictors " << DateTime() << "\n";
    SimulatePredictors(sample_size, total_predictor_dim);
    coefficients_ = Vector(predictors_.ncol(), 0.0);
    coefficients_[0] = 12;
    coefficients_[3] = -50;
    coefficients_[22] = 90;
    coefficients_[93] = 200;
    SimulateResponse();
    FillRegressionData();
    std::cout << "Done simulating data " << DateTime() << "\n";

    // std::cout << "Should see significance in positions 0, 3, 22, 93\n";

    NEW(BigRegressionModel, model)(total_predictor_dim, max_model_dim);
    std::cout << "streaming for initial screen: " << DateTime() << "\n";
    for (int i = 0; i < sample_size; ++i) {
      model->stream_data_for_initial_screen(*regression_data_[i]);
    }

    std::cout << "done streaming: " << DateTime() << "\n";
    Vector global_xty = response_ * predictors_;
    RegressionModel *m0 = model->subordinate_model(0);
    RegressionModel *m1 = model->subordinate_model(
        model->number_of_subordinate_models() - 1);

    EXPECT_TRUE(VectorEquals(ConstVectorView(global_xty, 0, m0->xdim()),
                             m0->suf()->xty(), 1e-5))
        << " differences: "
        << ConstVectorView(global_xty, 0, m0->xdim()) - m0->suf()->xty() << "\n";
    EXPECT_NEAR(global_xty.back(),
                m1->suf()->xty().back(),
                1e-5);

    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(total_predictor_dim,
                                       1.0 / total_predictor_dim);
    NEW(RegressionSlabPrior, slab_prototype)(
        SpdMatrix(1.0),
        model->Sigsq_prm(),
        1.0,
        1.0,
        5.0,
        0.1);

    NEW(BigAssSpikeSlabSampler, sampler)(
        model.get(), spike, slab_prototype, residual_precision_prior);
    model->set_method(sampler);

    bool use_threads = false;
    double inclusion_threshold = .10;
    std::cout << "beginning initial screen: " << DateTime() << "\n";
    sampler->initial_screen(100, inclusion_threshold, use_threads);
    std::cout << "done with initial screen: " << DateTime() << "\n";

    Selector good_ones(std::vector<int>{0, 3, 22, 93}, total_predictor_dim);
    EXPECT_EQ(good_ones, model->candidate_selector());

    for (int i = 0; i < sample_size; ++i) {
      model->stream_data_for_restricted_model(*regression_data_[i]);
    }

    int niter = 100;
    Vector sigma_draws(niter);
    Matrix beta_draws(niter, total_predictor_dim);
    std::cout << "beginning draws for compressed model: " << DateTime() << std::endl;
    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
    }
    std::cout << "done with draws for compressed model: " << DateTime() << std::endl;

    for (int el : std::vector<int>{0, 3, 22, 93}) {
      const ConstVectorView beta(beta_draws.col(el));
      EXPECT_GE(inclusion_probability(beta), .90) << SubMatrix(beta_draws, 90, 99, 0, 8);
    }
  }


}  // namespace

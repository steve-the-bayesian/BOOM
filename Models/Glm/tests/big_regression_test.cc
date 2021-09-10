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

#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class BigRegressionTest : public ::testing::Test {
   protected:

    BigRegressionTest()
        : residual_sd_(0.3)
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

  inline double inclusion_probability(const ConstVectorView &coefficients) {
    double ans = 0;
    for (auto y : coefficients) {
      ans += (y != 0);
    }
    return ans / coefficients.size();
  }

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

  TEST_F(BigRegressionTest, EndToEnd) {
    int max_model_dim = 100;
    int total_predictor_dim = 1000;
    int nonzero_predictor_dim = 50;
    int sample_size = 1000;
    SimulatePredictors(sample_size, total_predictor_dim);
    SimulateCoefficients(nonzero_predictor_dim);
    SimulateResponse();
    FillRegressionData();

    NEW(BigRegressionModel, model)(total_predictor_dim, max_model_dim);
    for (int i = 0; i < sample_size; ++i) {
      model->stream_data_for_initial_screen(*regression_data_[i]);
    }

    NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
    NEW(VariableSelectionPrior, spike)(total_predictor_dim, .5);
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
    sampler->initial_screen(100, .10);

    // for (int i = 0; i < sample_size; ++i) {
    //   model->stream_data_for_restricted_model(*regression_data_[i]);
    // }

    int niter = 100;
    Vector sigma_draws(niter);
    Matrix beta_draws(niter, total_predictor_dim);
    for (int i = 0; i < niter; ++i) {
      ////////////////////      model->sample_posterior();
      sigma_draws[i] = model->sigma();
      beta_draws.row(i) = model->Beta();
    }

    // NEW(RegressionModel, model2)(predictors_, response_);
    // NEW(MvnGivenScalarSigma, slab2)(
    //     Vector(xdim_, 0), xtx / nobs_, model2->Sigsq_prm());
    // NEW(BregVsSampler, old_sampler)(
    //     model2.get(), slab2, residual_precision_prior, spike);
    // model2->set_method(old_sampler);

    // Vector more_sigma_draws(niter_);
    // Matrix more_beta_draws(niter_, xdim_);
    // for (int i = 0; i < niter_; ++i) {
    //   model2->sample_posterior();
    //   more_sigma_draws[i] = model2->sigma();
    //   more_beta_draws.row(i) = model2->Beta();
    // }

    // EXPECT_TRUE(EquivalentSimulations(sigma_draws, more_sigma_draws))
    //     << cbind(sigma_draws, more_sigma_draws);

    // EXPECT_TRUE(EquivalentSimulations(beta_draws.col(3),
    //                                   more_beta_draws.col(3)))
    //     << cbind(beta_draws, more_beta_draws)
    //     << AsciiDistributionCompare(beta_draws.col(3),
    //                                 more_beta_draws.col(3));

    // for (int i = 0; i < xdim_; ++i) {
    //   EXPECT_TRUE(EquivalentSimulations(beta_draws.col(i),
    //                                     more_beta_draws.col(i)));
    // }
  }

  // TEST_F(RegressionSpikeSlabTest, Large) {
  //   xdim_ = 100;
  //   SimulatePredictors();
  //   SimulateCoefficients();
  //   SimulateResponse();

  //   NEW(RegressionModel, model)(predictors_, response_, false);
  //   model->coef().drop_all();
  //   model->coef().add(0);

  //   SpdMatrix xtx = model->suf()->xtx();
  //   NEW(MvnGivenScalarSigma, slab)(
  //       Vector(xdim_, 0), xtx / nobs_, model->Sigsq_prm());
  //   NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
  //   NEW(VariableSelectionPrior, spike)(xdim_, 5.0 / xdim_);
  //   NEW(AdaptiveSpikeSlabRegressionSampler, sampler)(
  //       model.get(), slab, residual_precision_prior, spike);
  //   model->set_method(sampler);
  //   Vector sigma_draws(niter_);
  //   Matrix beta_draws(niter_, xdim_);
  //   Vector size_draws(niter_);
  //   for (int i = 0; i < niter_; ++i) {
  //     model->sample_posterior();
  //     sigma_draws[i] = model->sigma();
  //     beta_draws.row(i) = model->Beta();
  //     size_draws[i] = model->coef().inc().nvars();
  //   }

  //   ECDF size_distribution(size_draws);
  //   // The fraction of models with 8 or fewer included predictors should be
  //   // high.
  //   EXPECT_GE(size_distribution(8), .95);
  // }

  // TEST_F(RegressionSpikeSlabTest, PerfectCollinearity) {
  //   xdim_ = 50;
  //   niter_ *= 5;
  //   SimulatePredictors();
  //   predictors_.col(2) = 1.7 * predictors_.col(1);
  //   predictors_.col(3) = -2.4 * predictors_.col(2);

  //   // Pick some obvious values for coefficients that the model should be able
  //   // to find easily.  Because of the perfect correlation, coefficient 1, 2,
  //   // and 3 each have about a 1/3 chance of appearing.
  //   coefficients_ = Vector(xdim_, 0.0);
  //   coefficients_[0] = 12;
  //   coefficients_[1] = 18;
  //   coefficients_[2] = 0;
  //   coefficients_[3] = 0;
  //   coefficients_[4] = -4;
  //   coefficients_[5] = 73;

  //   SimulateResponse();

  //   NEW(RegressionModel, model)(predictors_, response_, false);
  //   model->coef().drop_all();
  //   model->coef().add(0);
  //   SpdMatrix xtx = model->suf()->xtx();
  //   NEW(MvnGivenScalarSigma, slab)(
  //       Vector(xdim_, 0), xtx / nobs_, model->Sigsq_prm());
  //   NEW(ChisqModel, residual_precision_prior)(1.0, 1.0);
  //   NEW(VariableSelectionPrior, spike)(xdim_, 5.0 / xdim_);
  //   NEW(BregVsSampler, sampler)(
  //       model.get(), slab, residual_precision_prior, spike);
  //   model->set_method(sampler);

  //   Matrix beta_draws(niter_, xdim_);
  //   Vector sigma_draws(niter_);

  //   for (int i = 0; i < niter_; ++i) {
  //     model->sample_posterior();
  //     sigma_draws[i] = model->sigma();
  //     beta_draws.row(i) = model->Beta();
  //   }

  //   EXPECT_GE(inclusion_probability(beta_draws.col(0)), .9);
  //   EXPECT_GE(inclusion_probability(beta_draws.col(1)), .20);
  //   EXPECT_LE(inclusion_probability(beta_draws.col(1)), .40);
  //   EXPECT_GE(inclusion_probability(beta_draws.col(2)), .20);
  //   EXPECT_LE(inclusion_probability(beta_draws.col(2)), .40);
  //   EXPECT_GE(inclusion_probability(beta_draws.col(3)), .20);
  //   EXPECT_LE(inclusion_probability(beta_draws.col(3)), .40);
  //   EXPECT_GE(inclusion_probability(beta_draws.col(4)), .9);
  //   EXPECT_GE(inclusion_probability(beta_draws.col(5)), .9);
  // }

}  // namespace

#include "gtest/gtest.h"
#include "distributions.hpp"

#include "Models/Mixtures/BetaBinomialMixture.hpp"
#include "Models/Mixtures/PosteriorSamplers/BetaBinomialMixturePosteriorSampler.hpp"

#include "Models/MultinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/UniformModel.hpp"

#include "Models/PosteriorSamplers/BetaBinomialPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"

#include "stats/logit.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class BetaBinomialMixtureTest : public ::testing::Test {
   protected:
    BetaBinomialMixtureTest() {
      GlobalRng::rng.seed(8675309);
      true_pi_hi = .3;
      true_a_lo = 300.0;
      true_b_lo = 700.0;
      true_a_hi = 1300.0;
      true_b_hi = 400.0;
      sample_size = 100;
      niter = 200;
    }

    void fill_model_components() {
      mixing_distribution.reset(new MultinomialModel(Vector{.5, .5}));
      mixing_distribution_prior.reset(new DirichletModel(Vector{1.0, 1.0}));
      NEW(MultinomialDirichletSampler, mixing_distribution_sampler)(
          mixing_distribution.get(), mixing_distribution_prior);
      mixing_distribution->set_method(mixing_distribution_sampler);

      mixture_components.push_back(new BetaBinomialModel(1.0, 2.0));
      NEW(BetaModel, lower_component_mean_prior)(1.0, 10.0);
      component_mean_priors.push_back(lower_component_mean_prior);
      NEW(UniformModel, lower_component_sample_size_prior)(0.1, 10000.0);
      sample_size_priors.push_back(lower_component_sample_size_prior);
      NEW(BetaBinomialPosteriorSampler, lower_component_sampler)(
          mixture_components.back().get(),
          lower_component_mean_prior,
          lower_component_sample_size_prior);
      mixture_components.back()->set_method(lower_component_sampler);

      mixture_components.push_back(new BetaBinomialModel(2.0, 1.0));
      NEW(BetaModel, upper_component_mean_prior)(10.0, 1.0);
      component_mean_priors.push_back(upper_component_mean_prior);
      NEW(UniformModel, upper_component_sample_size_prior)(0.1, 10000.0);
      sample_size_priors.push_back(upper_component_sample_size_prior);
      NEW(BetaBinomialPosteriorSampler, upper_component_sampler)(
          mixture_components.back().get(),
          upper_component_mean_prior,
          upper_component_sample_size_prior);
      mixture_components.back()->set_method(upper_component_sampler);

      model = new BetaBinomialMixtureModel(mixture_components,
                                           mixing_distribution);
    }

    void simulate_data(double pi, double plo, double nulo, double phi, double nuhi, int sample_size_arg) {
      true_a_lo = plo * nulo;
      true_a_hi = phi * nuhi;
      true_b_lo = nulo - true_a_lo;
      true_b_hi = nuhi - true_a_hi;
      true_pi_hi = pi;
      sample_size = sample_size_arg;

      for (int i = 0; i < sample_size; ++i) {
        int N = rpois(1000);
        int hi = runif() < true_pi_hi;

        double p = hi ? rbeta(true_a_hi, true_b_hi) : rbeta(true_a_lo, true_b_lo);
        int y = rbinom(N, p);
        p_hat.push_back(y * 1.0 / N);
        suf.add_data(N, y, 1);
      }

      for (const auto &el : suf.count_table()) {
        NEW(AggregatedBinomialData, data_point)(el.first.first, el.first.second, el.second);
        model->add_data(data_point);
      }

    }

    double true_pi_hi;
    double true_a_lo;
    double true_b_lo;
    double true_a_hi;
    double true_b_hi;
    int sample_size;
    int niter;

    Ptr<MultinomialModel> mixing_distribution;
    Ptr<DirichletModel> mixing_distribution_prior;
    std::vector<Ptr<BetaBinomialModel>> mixture_components;
    std::vector<Ptr<BetaModel>> component_mean_priors;
    std::vector<Ptr<DoubleModel>> sample_size_priors;
    Ptr<BetaBinomialMixtureModel> model;

    BetaBinomialSuf suf;
    Vector p_hat;
  };

  TEST_F(BetaBinomialMixtureTest, BasicModelFunctions) {
    NEW(MultinomialModel, mixing_distribution)(Vector{.3, .2, .5});
    std::vector<Ptr<BetaBinomialModel>> mixture_components;
    mixture_components.push_back(new BetaBinomialModel(1.0, 2.0));
    mixture_components.push_back(new BetaBinomialModel(2.0, 1.0));
    mixture_components.push_back(new BetaBinomialModel(15.0, 45.0));
    BetaBinomialMixtureModel model(mixture_components, mixing_distribution);

    EXPECT_EQ(3, model.number_of_mixture_components());
  }

  TEST_F(BetaBinomialMixtureTest, LogLikelihoodTest) {
    NEW(MultinomialModel, mixing_distribution)(Vector{.3, .2, .5});
    std::vector<Ptr<BetaBinomialModel>> mixture_components;
    mixture_components.push_back(new BetaBinomialModel(1.0, 2.0));
    mixture_components.push_back(new BetaBinomialModel(2.0, 1.0));
    mixture_components.push_back(new BetaBinomialModel(15.0, 45.0));
    BetaBinomialMixtureModel model(mixture_components, mixing_distribution);

    model.add_data(new AggregatedBinomialData(10, 4, 3));
    model.add_data(new AggregatedBinomialData(12, 2, 1));

    Vector weights = {.2, .1, .7};
    Vector a = {1.0, 2.0, 5.0};
    Vector b = {8.0, 6.0, 7.0};
    Matrix ab = cbind(a, b);
    double loglike_model = model.log_likelihood(weights, ab);

    Vector f0;
    f0.push_back(exp(BetaBinomialModel::logp(10, 4, 1.0, 8.0)));
    f0.push_back(exp(BetaBinomialModel::logp(10, 4, 2.0, 6.0)));
    f0.push_back(exp(BetaBinomialModel::logp(10, 4, 5.0, 7.0)));
    double logp0 = 3 * log(weights.dot(f0));

    Vector f1;
    f1.push_back(exp(BetaBinomialModel::logp(12, 2, 1.0, 8.0)));
    f1.push_back(exp(BetaBinomialModel::logp(12, 2, 2.0, 6.0)));
    f1.push_back(exp(BetaBinomialModel::logp(12, 2, 5.0, 7.0)));
    double logp1 = log(weights.dot(f1));

    double loglike_direct = logp0 + logp1;
    EXPECT_NEAR(loglike_direct, loglike_model, 1e-6);
  }

  void swap_components(Ptr<BetaBinomialMixtureModel> model) {
    double pi = model->mixing_distribution()->pi()[1];
    double alo = model->mixture_component(0)->a();
    double ahi = model->mixture_component(1)->a();
    double blo = model->mixture_component(0)->b();
    double bhi = model->mixture_component(1)->b();

    model->mixing_distribution()->set_pi(Vector{1-pi, pi});
    model->mixture_component(0)->set_a(ahi);
    model->mixture_component(0)->set_b(bhi);
    model->mixture_component(1)->set_a(alo);
    model->mixture_component(1)->set_b(blo);
  }

  // Given an easy data set, with very strong separation between the low and
  // high modes, check to see that model parameters are recovered.
  TEST_F(BetaBinomialMixtureTest, McmcTest) {
    fill_model_components();
    NEW(BetaBinomialMixturePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    simulate_data(.3, .3, 1000, 13.0 / 17, 1700, 200);

    Vector pi_draws(niter);
    Vector p_lo_draws(niter);
    Vector p_hi_draws(niter);
    Vector nu_lo_draws(niter);
    Vector nu_hi_draws(niter);

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      if (model->mixture_component(1)->mean() < model->mixture_component(0)->mean()) {
        swap_components(model);
      }
      pi_draws[i] = model->mixing_distribution()->pi()[1];
      p_lo_draws[i] = model->mixture_component(0)->mean();
      p_hi_draws[i] = model->mixture_component(1)->mean();
      nu_lo_draws[i] = model->mixture_component(0)->prior_sample_size();
      nu_hi_draws[i] = model->mixture_component(1)->prior_sample_size();
    }

    double true_p_lo = true_a_lo / (true_a_lo + true_b_lo);
    double true_p_hi = true_a_hi / (true_a_hi + true_b_hi);
    double true_nu_lo = true_a_lo + true_b_lo;
    double true_nu_hi = true_a_hi + true_b_hi;

    // Set this flag to 'true' to force parameter output to files.
    bool force_output = false;
    EXPECT_TRUE(CheckMcmcVector(pi_draws, true_pi_hi, 0.95, "pi.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(p_lo_draws, true_p_lo, 0.95, "p_lo.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(p_hi_draws, true_p_hi, 0.95, "p_hi.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(nu_lo_draws, true_nu_lo, 0.95, "nu_lo.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(nu_hi_draws, true_nu_hi, 0.95, "nu_hi.draws", force_output));
    // std::ofstream("p_hat.data") << p_hat;
  }

  //===========================================================================
  TEST_F(BetaBinomialMixtureTest, DirectMcmcTest) {
    fill_model_components();
    NEW(BetaBinomialMixtureDirectPosteriorSampler, sampler)(
        model.get(),
        mixing_distribution_prior, component_mean_priors,
        sample_size_priors);
    model->set_method(sampler);

    simulate_data(.1, .01, 100, .99, 200, 300);
    Vector pi_draws(niter);
    Vector p_lo_draws(niter);
    Vector p_hi_draws(niter);
    Vector nu_lo_draws(niter);
    Vector nu_hi_draws(niter);

    for (int i = 0; i < niter; ++i) {
      //std::cout << "----------------- iteration " << i << "\n";
      model->sample_posterior();
      if (model->mixture_component(1)->mean() < model->mixture_component(0)->mean()) {
        swap_components(model);
      }

      Vector theta = sampler->pack_theta();
      Vector mixing_weight_values, prob_values, sample_size_values;
      sampler->unpack_theta(theta, mixing_weight_values, prob_values, sample_size_values);
      EXPECT_DOUBLE_EQ(theta[0], log(mixing_weight_values[1] / mixing_weight_values[0]));

      EXPECT_NEAR(theta[1], logit(prob_values[0]), 1e-6);
      EXPECT_NEAR(theta[2], log(sample_size_values[0]), 1e-6);
      EXPECT_NEAR(theta[3], logit(prob_values[1]), 1e-6);
      EXPECT_NEAR(theta[4], log(sample_size_values[1]), 1e-6);

      pi_draws[i] = model->mixing_distribution()->pi()[1];
      //std::cout << model->mixing_distribution()->pi() << "\n";

      p_lo_draws[i] = model->mixture_component(0)->mean();
      p_hi_draws[i] = model->mixture_component(1)->mean();
      nu_lo_draws[i] = model->mixture_component(0)->prior_sample_size();
      nu_hi_draws[i] = model->mixture_component(1)->prior_sample_size();

      // std::cout << "pi    = " << pi_draws[i] << "\n"
      //           << "p_lo   = " << p_lo_draws[i] << "\n"
      //           << "nu_lo = " << nu_lo_draws[i] << "\n"
      //           << "p_hi   = " << p_hi_draws[i] << "\n"
      //           << "nu_hi = " << nu_hi_draws[i] << "\n";
    }

    double true_p_lo = true_a_lo / (true_a_lo + true_b_lo);
    double true_p_hi = true_a_hi / (true_a_hi + true_b_hi);
    double true_nu_lo = true_a_lo + true_b_lo;
    double true_nu_hi = true_a_hi + true_b_hi;

    // Set this flag to 'true' to force parameter output to files.
    bool force_output = false;
    EXPECT_TRUE(CheckMcmcVector(pi_draws, true_pi_hi, 0.95, "direct_pi.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(p_lo_draws, true_p_lo, 0.95, "direct_p_lo.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(p_hi_draws, true_p_hi, 0.95, "direct_p_hi.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(nu_lo_draws, true_nu_lo, 0.95, "direct_nu_lo.draws", force_output));
    EXPECT_TRUE(CheckMcmcVector(nu_hi_draws, true_nu_hi, 0.95, "direct_nu_hi.draws", force_output));
    // std::ofstream("p_hat.data") << p_hat;
  }

}  // namespace

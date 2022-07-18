#include "gtest/gtest.h"
#include "distributions.hpp"

#include "Models/Mixtures/BetaBinomialMixture.hpp"
#include "Models/Mixtures/PosteriorSamplers/BetaBinomialMixturePosteriorSampler.hpp"

#include "Models/MultinomialModel.hpp"
#include "Models/BetaModel.hpp"
#include "Models/UniformModel.hpp"

#include "Models/PosteriorSamplers/BetaBinomialPosteriorSampler.hpp"
#include "Models/PosteriorSamplers/MultinomialDirichletSampler.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;

  class BetaBinomialMixtureTest : public ::testing::Test {
   protected:
    BetaBinomialMixtureTest() {
      GlobalRng::rng.seed(8675309);
    }
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
    double true_pi_hi = .3;
    double true_a_lo = 300.0;
    double true_b_lo = 700.0;
    double true_a_hi = 1300.0;
    double true_b_hi = 400.0;
    int sample_size = 100;
    int niter = 200;

    NEW(MultinomialModel, mixing_distribution)(Vector{.5, .5});
    NEW(MultinomialDirichletSampler, mixing_distribution_sampler)(
        mixing_distribution.get(), Vector{1.0, 1.0});
    mixing_distribution->set_method(mixing_distribution_sampler);

    std::vector<Ptr<BetaBinomialModel>> mixture_components;
    mixture_components.push_back(new BetaBinomialModel(1.0, 2.0));
    NEW(BetaModel, lower_component_mean_prior)(1.0, 10.0);
    NEW(UniformModel, lower_component_sample_size_prior)(0.1, 10000.0);
    NEW(BetaBinomialPosteriorSampler, lower_component_sampler)(
        mixture_components.back().get(),
        lower_component_mean_prior,
        lower_component_sample_size_prior);
    mixture_components.back()->set_method(lower_component_sampler);

    mixture_components.push_back(new BetaBinomialModel(2.0, 1.0));
    NEW(BetaModel, upper_component_mean_prior)(10.0, 1.0);
    NEW(UniformModel, upper_component_sample_size_prior)(0.1, 10000.0);
    NEW(BetaBinomialPosteriorSampler, upper_component_sampler)(
        mixture_components.back().get(),
        upper_component_mean_prior,
        upper_component_sample_size_prior);
    mixture_components.back()->set_method(upper_component_sampler);

    NEW(BetaBinomialMixtureModel, model)(mixture_components, mixing_distribution);
    NEW(BetaBinomialMixturePosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    BetaBinomialSuf suf;
    Vector p_hat;
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

}  // namespace

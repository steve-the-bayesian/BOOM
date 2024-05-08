#include "gtest/gtest.h"
#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/FactorModels/PosteriorSamplers/PoissonFactorModelIndependentGammaPosteriorSampler.hpp"
#include "Models/FactorModels/PosteriorSamplers/PoissonFactorHierarchicalSampler.hpp"

#include "Models/GammaModel.hpp"
#include "distributions.hpp"
#include "TargetFun/SumMultinomialLogitTransform.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>
#include <string>

namespace {
  using namespace BOOM;
  using Visitor = BOOM::FactorModels::PoissonVisitor;
  using Site = BOOM::FactorModels::PoissonSite;
  using std::endl;
  using std::cout;

  class PoissonFactorModelTest : public ::testing::Test {
   protected:
    PoissonFactorModelTest() {
      GlobalRng::rng.seed(8675309);
    }

    // Args:
    //   site_lambdas: Each row represents a site, and each column a latent
    //     category.
    //   class_indicators:  Each element is a class indicator for a visitor.
    //
    // Returns:
    //   A data set containing visits to sites.
    std::vector<Ptr<PoissonFactorData>> simulate_data(
        const Matrix &site_lambdas,
        std::vector<int> class_indicators) {
      std::vector<Ptr<PoissonFactorData>> ans;
      for (int i = 0; i < class_indicators.size(); ++i) {
        for (int j = 0; j < site_lambdas.nrow(); ++j) {
          int nvisits = rpois(site_lambdas(j, class_indicators[i]));
          if (nvisits > 0) {
            NEW(PoissonFactorData, data_point)(std::to_string(i),
                                               std::to_string(j),
                                               nvisits);
            ans.push_back(data_point);
          }
        }
      }
      return ans;
    }
  };

  std::ostream & operator<<(std::ostream &out, const std::vector<int> &vec) {
    for (int val : vec) {
      out << std::setw(10) << val;
    }
    return out;
  }

  TEST_F(PoissonFactorModelTest, SmokeTest) {
    PoissonFactorModel model(3);
  }

  TEST_F(PoissonFactorModelTest, VisitorTest) {
    Visitor visitor("Larry", 2);
    EXPECT_EQ(visitor.id(), "Larry");

    NEW(Site, site1)("8", 2);
    visitor.visit(site1, 4);

    EXPECT_EQ(visitor.sites_visited().size(), 1);

    NEW(Site, site2)("6", 2);
    visitor.visit(site2, 1);
    EXPECT_EQ(visitor.sites_visited().size(), 2);

    visitor.set_class_probabilities(Vector{.4, .6});
    EXPECT_TRUE(VectorEquals(visitor.class_probabilities(), Vector{.4, .6}));

    EXPECT_EQ(visitor.imputed_class_membership(), -1);
    visitor.set_class_member_indicator(1);
    EXPECT_EQ(visitor.imputed_class_membership(), 1);
  }

  // Check that the
  TEST_F(PoissonFactorModelTest, SiteTest) {
    Site site("867", 4);

    EXPECT_EQ("867", site.id());

    EXPECT_EQ(site.lambda().size(), 4);

    NEW(Visitor, v1)("Larry", 4);
    NEW(Visitor, v2)("Moe", 4);
    site.observe_visitor(v1, 1);
    EXPECT_EQ(site.observed_visitors().size(), 1);
    site.observe_visitor(v2, 1);
    EXPECT_EQ(site.observed_visitors().size(), 2);

    // Observing a previously observed visitor does not result in a new
    // visitor.
    site.observe_visitor(v2, 20);
    EXPECT_EQ(site.observed_visitors().size(), 2);

    EXPECT_EQ(site.observed_visitors().find(v2)->second, 21);
  }

  TEST_F(PoissonFactorModelTest, ModelTest) {
    int num_classes = 4;
    int num_visitors = 1000;
    int num_sites = 20;
    int niter = 100;
    PoissonFactorModel model(num_classes);

    EXPECT_EQ(num_classes, model.number_of_classes());

    Vector class_membership_probabilities = {0.1, 0.3, 0.5, 0.1};

    EXPECT_EQ(num_classes, class_membership_probabilities.size());

    Matrix site_lambdas(num_sites, num_classes);
    for (int i = 0; i < num_sites; ++i) {
      for (int j = 0; j < num_classes; ++j){
        site_lambdas(i, j) = rgamma(0.8, 2.0);
      }
    }

    std::vector<int> class_indicators = rmulti_vector_mt(
        GlobalRng::rng, num_visitors, class_membership_probabilities);

    std::vector<Ptr<PoissonFactorData>> data = simulate_data(
        site_lambdas, class_indicators);

    for (const auto &data_point : data) {
      model.add_data(data_point);
    }

    EXPECT_EQ(model.sites().size(), num_sites);
    EXPECT_EQ(model.visitors().size(), num_visitors);

    EXPECT_EQ(model.site("12")->id(), "12");
    EXPECT_EQ(model.visitor("12")->id(), "12");

    std::vector<Ptr<GammaModelBase>> default_intensity_prior;
    for (int i = 0; i < 4; ++i) {
      default_intensity_prior.push_back(new GammaModel(.8, 2.0));
    }

    NEW(PoissonFactorModelIndependentGammaPosteriorSampler, sampler)(
        &model,
        class_membership_probabilities,
        default_intensity_prior);
    model.set_method(sampler);

    Matrix visitor_draws(niter, num_visitors);
    //     Array lambda_draws({niter, num_sites, num_classes});
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      for (int j = 0; j < num_visitors; ++j) {
        visitor_draws(i, j) = model.visitor(std::to_string(j))->imputed_class_membership();
      }
    }
    std::ofstream visitor_out("visitor_draws.out");
    visitor_out << class_indicators << "\n";
    visitor_out << visitor_draws;
  }

  // Given a set of visitors with known categories, check that the posterior
  // recovers the true values.
  //
  // (1) all categories observed with large sample sizes
  // (2) Many sites, with some elements missing from each site.
  // (3) All categories but one highly observed.  Remaining category sparsely
  //     observed.
  TEST_F(PoissonFactorModelTest, SiteLogPosteriorTest) {
    //-------------------------------------------------------------------------
    // Define test parameters and simulate fake data.
    //-------------------------------------------------------------------------
    int num_users = 200;
    int num_sites = 50;
    int niter = 100;

    Vector class_probs = {.05, .25, .7};
    std::vector<int> user_categories;
    for (int i = 0; i < num_users; ++i) {
      user_categories.push_back(rmulti(class_probs));
    }

    Vector multinomial_logit_mean = {-.2, .7};
    SpdMatrix multinomial_logit_variance(Vector{.4, .1, .1, .8});
    Matrix eta(num_sites, 3);
    Matrix lambda(num_sites, 3);
    SumMultinomialLogitTransform transform;
    for (int i = 0; i < num_sites; ++i) {
      eta(i, 0) = runif(0, 10);
      VectorView(eta.row(i), 1) = rmvn(
          multinomial_logit_mean, multinomial_logit_variance);
      lambda.row(i) = transform.from_sum_logits(eta.row(i));
    }

    std::vector<Ptr<PoissonFactorData>> data = simulate_data(
        lambda, user_categories);

    //-------------------------------------------------------------------------
    // Create the model and prior.
    //-------------------------------------------------------------------------
    NEW(PoissonFactorModel, model)(3);
    for (const auto &dp : data) model->add_data(dp);

    Vector default_class_membership_probabilities(3, 1.0 / 3);
    Vector multinomial_logit_prior_mean(2, 0.0);
    SpdMatrix multinomial_logit_prior_variance(2, 1.0);

    NEW(PoissonFactorHierarchicalSampler, sampler)(
        model.get(),
        default_class_membership_probabilities,
        multinomial_logit_prior_mean,
        1.0,
        multinomial_logit_prior_variance,
        3.0);
    model->set_method(sampler);

    //-------------------------------------------------------------------------
    // Fix the user data at known values.
    //-------------------------------------------------------------------------
    std::vector<Vector> known_class_priors;
    known_class_priors.push_back(Vector{1.0, 0, 0});
    known_class_priors.push_back(Vector{0, 1.0, 0});
    known_class_priors.push_back(Vector{0, 0, 1.0});
    for (const auto &dp : data) {
      int which_visitor = std::stoi(dp->visitor_id());
      sampler->set_prior_class_probabilities(
          dp->visitor_id(),
          known_class_priors[user_categories[which_visitor]]);
    }

    //-------------------------------------------------------------------------
    // Create space to hold the MCMC draws.
    //-------------------------------------------------------------------------
    Matrix mu_draws(niter, 2);
    Matrix Sigma_draws(niter, 4);
    //-------------------------------------------------------------------------
    // Run the MCMC
    //-------------------------------------------------------------------------

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      mu_draws.row(i) = sampler->hyperprior()->mu();
      Sigma_draws.row(i) = sampler->hyperprior()->Sigma().vectorize(false);
    }

    auto status = CheckMcmcMatrix(mu_draws, multinomial_logit_mean);
    EXPECT_TRUE(status.ok) << status;

    status = CheckMcmcMatrix(
        Sigma_draws,
        multinomial_logit_variance.vectorize(false));
    EXPECT_TRUE(status.ok) << status;

    // std::ofstream mu_draws_out("mu.draws");
    // mu_draws_out
    //     << multinomial_logit_mean << "\n"
    //     << mu_draws;

    // std::ofstream Sigma_draws_out("Sigma.draws");
    // Sigma_draws_out
    //     << multinomial_logit_variance.vectorize(false) << "\n"
    //     << Sigma_draws;
  }



}  // namespace

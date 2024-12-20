#include "gtest/gtest.h"
#include "Models/FactorModels/MultinomialFactorModel.hpp"
#include "Models/FactorModels/PosteriorSamplers/MultinomialFactorModelPosteriorSampler.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>
#include <string>

#include <random>
#include <algorithm>

namespace {
  using namespace BOOM;
  using Visitor = BOOM::FactorModels::MultinomialVisitor;
  using Site = BOOM::FactorModels::MultinomialSite;
  using std::endl;
  using std::cout;

  class MultinomialFactorModelTest : public ::testing::Test {
   protected:
    MultinomialFactorModelTest() {
      GlobalRng::rng.seed(8675309);
    }

    // Args:
    //   site_lambdas: Each row represents a site, and each column a latent
    //     category.
    //   class_indicators:  Each element is a class indicator for a visitor.
    //   other_class_name: The name of a default  to use when
    //
    // Returns:
    //   A data set containing visits to sites.
    std::vector<Ptr<MultinomialFactorData>> simulate_data(
        const Matrix &site_probs,

        std::vector<int> class_indicators,
        const std::string &default_site_name = "Other") {
      std::vector<Ptr<MultinomialFactorData>> ans;
      for (int i = 0; i < class_indicators.size(); ++i) {
        int Ntrials = rpois(8);
        for (int m = 0; m < Ntrials; ++m) {
          int which_site = rmulti(site_probs.col(class_indicators[i]));
          std::string site_name = std::to_string(which_site);
          if (which_site == site_probs.nrow() - 1) {
            site_name = "Other";
          }
          NEW(MultinomialFactorData, data_point)(
              std::to_string(i),
              site_name,
              1);
          ans.push_back(data_point);
        }
      }
      return ans;
    }
  };

  //===========================================================================
  TEST_F(MultinomialFactorModelTest, SmokeTest) {
    MultinomialFactorModel model(3);
  }

  //===========================================================================
  TEST_F(MultinomialFactorModelTest, VisitorTest) {
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

  //===========================================================================
  // Check that the
  TEST_F(MultinomialFactorModelTest, SiteTest) {
    Site site("867", 4);

    EXPECT_EQ("867", site.id());

    EXPECT_EQ(site.visit_probs().size(), 4);

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

  //===========================================================================
  TEST_F(MultinomialFactorModelTest, ModelTest) {
    int num_classes = 4;
    int num_visitors = 1000;
    int num_sites = 20;
    int niter = 100;
    MultinomialFactorModel model(num_classes);

    EXPECT_EQ(num_classes, model.number_of_classes());

    Vector class_membership_probabilities = {0.1, 0.3, 0.5, 0.1};

    EXPECT_EQ(num_classes, class_membership_probabilities.size());

    Matrix site_probs(num_sites, num_classes);
    for (int j = 0; j < num_classes; ++j){
      site_probs.col(j) = rdirichlet(Vector(num_sites, 1.0));
    }

    std::vector<int> class_indicators = rmulti_vector_mt(
        GlobalRng::rng, num_visitors, class_membership_probabilities);

    std::vector<Ptr<MultinomialFactorData>> data = simulate_data(
        site_probs, class_indicators, "Other");

    for (const auto &data_point : data) {
      model.add_data(data_point);
    }

    EXPECT_EQ(model.sites().size(), num_sites);
    EXPECT_EQ(model.visitors().size(), num_visitors);

    EXPECT_EQ(model.site("12")->id(), "12");
    EXPECT_EQ(model.visitor("12")->id(), "12");

    EXPECT_EQ(model.site("Other")->id(), "Other");

    NEW(MultinomialFactorModelPosteriorSampler, sampler)(
        &model,
        class_membership_probabilities);
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

    sampler->set_num_threads(8);
    for (int i = 0; i < niter; ++i) {
      model.sample_posterior();
      for (int j = 0; j < num_visitors; ++j) {
        visitor_draws(i, j) = model.visitor(std::to_string(j))->imputed_class_membership();
      }
    }
    std::ofstream threaded_visitor_out("visitor_draws_threaded.out");
    threaded_visitor_out << class_indicators << "\n";
    threaded_visitor_out << visitor_draws;
  }

  //===========================================================================
  TEST_F(MultinomialFactorModelTest, MergeTest) {
    int num_classes = 4;
    int num_visitors = 1000;
    int num_sites = 20;

    MultinomialFactorModel model1(num_classes);
    MultinomialFactorModel model2(num_classes);
    MultinomialFactorModel model3(num_classes);

    Vector class_membership_probabilities = {0.1, 0.3, 0.5, 0.1};
    Matrix site_probs(num_sites, num_classes);
    for (int j = 0; j < num_classes; ++j){
      site_probs.col(j) = rdirichlet(Vector(num_sites, 1.0));
    }

    std::vector<int> class_indicators = rmulti_vector_mt(
        GlobalRng::rng, num_visitors, class_membership_probabilities);

    // Simulate the data, and split it into two subsets.
    std::vector<Ptr<MultinomialFactorData>> data = simulate_data(
        site_probs, class_indicators);

    std::random_device rd;
    std::mt19937 randomizer(rd());
    std::shuffle(data.begin(), data.end(), randomizer);

    size_t sample_size = data.size();
    std::vector<Ptr<MultinomialFactorData>> subset1(
        data.begin(), data.begin() + sample_size / 2);
    std::vector<Ptr<MultinomialFactorData>> subset2(
        data.begin() + sample_size / 2, data.end());

    // Give the full data set to model1, partial subsets to the other two
    // models, then combine models 2 and 3.
    for (const auto &el : data) model1.add_data(el);
    for (const auto &el : subset1) model2.add_data(el);
    for (const auto &el : subset2) model3.add_data(el);
    model3.combine_data_mt(model2);

    // Check that the data stored in model1 is identical to that in model 3.

    // Step 1: check the visitors...
    auto visitors1 = model1.visitors();
    auto visitors3 = model3.visitors();
    EXPECT_EQ(visitors1.size(), visitors3.size());
    auto vit1 = visitors1.begin();
    auto vit3 = visitors3.begin();
    for (; vit1 != visitors1.end(); ++vit1, ++vit3) {
      EXPECT_EQ(vit1->first, vit3->first);

      // v1_sites is a map of all the sites visited by visitor shown in model 1.
      // v3_sites is a map of all the sites visited by visitor shown in model 3.
      auto v1_sites = vit1->second->sites_visited();
      auto v3_sites = vit3->second->sites_visited();
      EXPECT_EQ(v1_sites.size(), v3_sites.size());
      // Now iterate through each of the sites visited by each visitor and make
      // sure they're the same.
      auto v1_sites_it = v1_sites.begin();
      auto v3_sites_it = v3_sites.begin();
      for (; v1_sites_it != v1_sites.end(); ++v1_sites_it, ++v3_sites_it) {
        Ptr<Site> v1_site = v1_sites_it->first;
        Ptr<Site> v3_site = v3_sites_it->first;
        int v1_site_count = v1_sites_it->second;
        int v3_site_count = v3_sites_it->second;
        EXPECT_EQ(v1_site_count, v3_site_count);
        EXPECT_EQ(v1_site->id(), v3_site->id());
      }
    }

    // Step 2: check the sites...
    auto sites1 = model1.sites();
    auto sites3 = model3.sites();
    EXPECT_EQ(sites1.size(), sites3.size());
    auto sit1 = sites1.begin();
    auto sit3 = sites3.begin();
    for (; sit1 != sites1.end(); ++sit1, ++sit3) {
      // Check that the site id's are equal.
      EXPECT_EQ(sit1->first, sit3->first);

      // s1_visitors is all the visitor to the site from model 1.
      // s3_visitors is all the visitor to the site from model 1.
      auto s1_visitors = sit1->second->observed_visitors();
      auto s3_visitors = sit3->second->observed_visitors();
      EXPECT_EQ(s1_visitors.size(), s3_visitors.size());

      // Iterate through the visitors from each site, and their visit counts.
      auto s1_visitors_it = s1_visitors.begin();
      auto s3_visitors_it = s3_visitors.begin();
      for (; s1_visitors_it != s1_visitors.end();
           ++s1_visitors_it, ++s3_visitors_it) {
        Ptr<Visitor> s1_visitor = s1_visitors_it->first;
        Ptr<Visitor> s3_visitor = s3_visitors_it->first;
        EXPECT_EQ(s1_visitor->id(), s3_visitor->id());

        int s1_count = s1_visitors_it->second;
        int s3_count = s3_visitors_it->second;
        EXPECT_EQ(s1_count, s3_count);
      }
    }
  }


}  // namespace

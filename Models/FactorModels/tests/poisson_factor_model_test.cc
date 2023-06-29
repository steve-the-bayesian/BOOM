#include "gtest/gtest.h"
#include "Models/FactorModels/PoissonFactorModel.hpp"
#include "Models/FactorModels/PosteriorSamplers/PoissonFactorModelPosteriorSampler.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using BOOM::PoissonFactor::Visitor;
  using BOOM::PoissonFactor::Site;
  using std::endl;
  using std::cout;

  class PoissonFactorModelTest : public ::testing::Test {
   protected:
    PoissonFactorModelTest() {
      GlobalRng::rng.seed(8675309);
    }

    std::vector<Ptr<PoissonFactorData>> simulate_data(
        const Matrix &site_lambdas, std::vector<int> class_indicators) {
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
    EXPECT_EQ(site.prior_a().size(), 4);
    EXPECT_EQ(site.prior_b().size(), 4);

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

    NEW(PoissonFactorModelPosteriorSampler, sampler)(
        &model, class_membership_probabilities);
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


}  // namespace

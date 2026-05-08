#include "gtest/gtest.h"
#include "Models/MultilevelMultinomialModel.hpp"
#include "Models/PosteriorSamplers/MultilevelMultinomialPosteriorSampler.hpp"
#include "Models/MultinomialModel.hpp"
#include "Models/DirichletModel.hpp"

#include "distributions.hpp"

#include "test_utils/test_utils.hpp"
#include <fstream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  class MultilevelMultinomialTest : public ::testing::Test {
   protected:
    MultilevelMultinomialTest() {
      GlobalRng::rng.seed(8675309);

      taxonomy_.reset(new Taxonomy({
            "shopping/clothes/shoes",
            "shopping/clothes/pants",
            "shopping/food/meat",
            "shopping/food/veggies",
            "restaurants/fancy/steak",
            "restaurants/fancy/seafood",    // not a leaf!
            "restaurants/fancy/seafood/fish",
            "restaurants/fancy/seafood/lobster"
          }));
    }
    Ptr<Taxonomy> taxonomy_;
  };

  TEST_F(MultilevelMultinomialTest, AddData) {
    NEW(MultilevelMultinomialModel, model)(taxonomy_);

    std::vector<Ptr<MultilevelCategoricalData>> data;
    data.push_back(new MultilevelCategoricalData(taxonomy_, "shopping/clothes"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "shopping/clothes/shoes"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "shopping/clothes/shoes"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy/seafood/lobster"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy/seafood/lobster"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy/seafood/lobster"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy/seafood/lobster"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy/seafood/fish"));

    model->add_data(data[0]);
    std::cout << model->top_level_model()->suf() << std::endl;

    EXPECT_EQ(1, model->number_of_observations());
    for (int i = 1; i < data.size(); ++i) {
      model->add_data(data[i]);
    }

    EXPECT_EQ(9, model->number_of_observations());
    EXPECT_EQ(9, model->top_level_model()->number_of_observations());
    EXPECT_EQ(3, model->conditional_model(
        "shopping")->number_of_observations());
    EXPECT_EQ(2, model->conditional_model(
        "shopping/clothes")->number_of_observations());

    EXPECT_EQ(6, model->conditional_model(
        "restaurants")->number_of_observations());
    EXPECT_EQ(5, model->conditional_model(
        "restaurants/fancy")->number_of_observations());
    EXPECT_EQ(5, model->conditional_model(
        "restaurants/fancy/seafood")->number_of_observations());

    model->clear_data();
    EXPECT_EQ(model->number_of_observations(), 0);
    EXPECT_EQ(0, model->top_level_model()->number_of_observations());
    EXPECT_EQ(0, model->conditional_model(
        "shopping")->number_of_observations());
    EXPECT_EQ(0, model->conditional_model(
        "shopping/clothes")->number_of_observations());

    EXPECT_EQ(0, model->conditional_model(
        "restaurants")->number_of_observations());
    EXPECT_EQ(0, model->conditional_model(
        "restaurants/fancy")->number_of_observations());
    EXPECT_EQ(0, model->conditional_model(
        "restaurants/fancy/seafood")->number_of_observations());
  }

  TEST_F(MultilevelMultinomialTest, TestMcmc) {
    // With a large sample size and a small taxonomy, each taxonomy element is
    // going to be visited at least once.
    int sample_size = 1000;
    int niter = 100;

    NEW(MultilevelMultinomialModel, model)(taxonomy_);
    NEW(MultilevelMultinomialPosteriorSampler, sampler)(model.get());
    model->set_method(sampler);

    std::vector<std::string> leaves = taxonomy_->leaf_names();

    for (int i = 0; i < sample_size; ++i) {
      int which = rmulti(0, leaves.size() - 1);
      NEW(MultilevelCategoricalData, data_point)(taxonomy_, leaves[which]);
      model->add_data(data_point);
    }

    Matrix top_level_draws(niter, model->top_level_model()->dim());

    Vector seafood_counts = model->conditional_model("restaurants/fancy/seafood")->suf()->n();
    Matrix seafood_draws(niter, seafood_counts.size());

    for (int i = 0; i < niter; ++i) {
      model->sample_posterior();
      top_level_draws.row(i) = model->top_level_model()->pi();
      seafood_draws.row(i) = model->conditional_model("restaurants/fancy/seafood")->pi();
    }

    Vector top_level_counts = model->top_level_model()->suf()->n();
    Vector top_level_probs = top_level_counts / sum(top_level_counts);
    CheckMcmcMatrix(top_level_draws, top_level_probs);

    Vector seafood_probs = seafood_counts / sum(seafood_counts);
    CheckMcmcMatrix(seafood_draws, seafood_probs);
  }

}  // namespace

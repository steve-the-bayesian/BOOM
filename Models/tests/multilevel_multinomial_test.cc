#include "gtest/gtest.h"
#include "Models/MultilevelMultinomialModel.hpp"
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

      taxonomy_ = read_taxonomy({
          "shopping/clothes/shoes",
          "shopping/clothes/pants",
          "shopping/food/meat",
          "shopping/food/veggies",
          "restaurants/fancy/steak",
          "restaurants/fancy/seafood",    // not a leaf!
          "restaurants/fancy/seafood/fish",
          "restaurants/fancy/seafood/lobster"
        });
    }
    Ptr<Taxonomy> taxonomy_;
  };


  TEST_F(MultilevelMultinomialTest, Model) {
    NEW(MultilevelMultinomialModel, model)(taxonomy_);

    std::vector<Ptr<MultilevelCategoricalData>> data;
    data.push_back(new MultilevelCategoricalData(taxonomy_, "shopping/clothes"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "shopping/clothes/shoes"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "shopping/clothes/shoes"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy/fish"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy"));
    data.push_back(new MultilevelCategoricalData(taxonomy_, "restaurants/fancy/seafood/lobster"));

    model->add_data(data[0]);
    std::cout << model->top_level_model()->suf() << std::endl;

    EXPECT_EQ(1, model->number_of_observations());
    for (int i = 1; i < data.size(); ++i) {
      model->add_data(data[i]);
    }

    EXPECT_EQ(6, model->number_of_observations());
    EXPECT_EQ(6, model->top_level_model()->number_of_observations());
    EXPECT_EQ(3, model->conditional_model("shopping")->number_of_observations());
  }


}  // namespace

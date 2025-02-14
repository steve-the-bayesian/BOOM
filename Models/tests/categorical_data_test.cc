#include "gtest/gtest.h"

#include "Models/CategoricalData.hpp"
#include "distributions.hpp"

#include "test_utils/check_derivatives.hpp"
#include "test_utils/test_utils.hpp"

#include <iostream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  inline void print(const std::vector<std::string> &stuff) {
    for (const auto &el : stuff) {
      std::cout << el << std::endl;
    }
  }
  
  class CatKeyTest : public ::testing::Test {
   protected:
    CatKeyTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(CatKeyTest, blah) {
  }

  class CategoricalDataTest : public ::testing::Test {
   protected:
    CategoricalDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };
  
  class TaxonomyTest : public ::testing::Test {
   protected:
    TaxonomyTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(TaxonomyTest, read_taxonomy_test) {
    std::vector<std::string> input_tax = {
      "shopping/clothes/shoes",
      "shopping/clothes/pants",
      "shopping/food/meat",
      "shopping/food/veggies",
      "restaurants/fancy/steak",
      "restaurants/fancy/seafood",    // not a leaf!
      "restaurants/fancy/seafood/fish",
      "restaurants/fancy/seafood/lobster"
    };

    Ptr<Taxonomy> tax = read_taxonomy(input_tax);
    std::cout << "leaf levels: \n";
    print(tax->leaf_names());

    std::cout << "full tree: \n";
    print(tax->node_names());
    
    EXPECT_EQ(tax->number_of_leaves(), 7);
    EXPECT_EQ(tax->tree_size(), 13);

    // When you add a taxonomy node that is already present, nothing happens.
    tax->add(std::vector<std::string>{"shopping"});
    EXPECT_EQ(tax->number_of_leaves(), 7);
    EXPECT_EQ(tax->tree_size(), 13);
  }

  TEST_F(TaxonomyTest, TestIteration) {
    std::vector<std::string> input_tax = {
      "shopping/clothes/shoes",
      "shopping/clothes/pants",
      "shopping/food/meat",
      "shopping/food/veggies",
      "restaurants/fancy/steak",
      "restaurants/fancy/seafood",    // not a leaf!
      "restaurants/fancy/seafood/fish",
      "restaurants/fancy/seafood/lobster"
    };

    Ptr<Taxonomy> tax = read_taxonomy(input_tax);
    std::vector<std::string> output;
    for (TaxonomyIterator it = tax->begin(); it != tax->end(); ++it) {
      output.push_back(it->path_from_root());
    }

    std::cout << "input_tax: \n";
    print(input_tax);

    std::cout << "output: \n";
    print(output);
    
  }
  
  class MultilevelCategoricalDataTest : public ::testing::Test {
   protected:
    MultilevelCategoricalDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(MultilevelCategoricalDataTest, TestGetSet) {
    std::vector<std::string> input_tax = {
      "shopping/clothes/shoes",
      "shopping/clothes/pants",
      "shopping/food/meat",
      "shopping/food/veggies",
      "restaurants/fancy/steak",
      "restaurants/fancy/seafood",    // not a leaf!
      "restaurants/fancy/seafood/fish",
      "restaurants/fancy/seafood/lobster"
    };

    Ptr<Taxonomy> tax = read_taxonomy(input_tax);
    std::string level = "restaurants/fancy/seafood/fish";
    
    NEW(MultilevelCategoricalData, data_point)(
        tax, level);
    std::vector<std::string> levels = {
      "restaurants", "fancy", "seafood", "fish"
    };
    NEW(MultilevelCategoricalData, dp2)(tax, levels);

    EXPECT_EQ(data_point->levels().size(), 4);

    EXPECT_EQ(dp2->levels().size(), 4);
    EXPECT_EQ(dp2->levels().size(), data_point->levels().size());
    EXPECT_EQ(dp2->levels()[0], data_point->levels()[0]);
    EXPECT_EQ(dp2->levels()[1], data_point->levels()[1]);
    EXPECT_EQ(dp2->levels()[2], data_point->levels()[2]);
    EXPECT_EQ(dp2->levels()[3], data_point->levels()[3]);

    EXPECT_EQ(dp2->name(), data_point->name());
    EXPECT_EQ(dp2->name(), level);
    
  }

}  // namespace

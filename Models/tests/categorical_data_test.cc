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
  
  class MultilevelCategoricalDataTest : public ::testing::Test {
   protected:
    MultilevelCategoricalDataTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

}  // namespace

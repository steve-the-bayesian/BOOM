#include "gtest/gtest.h"

#include "Models/CategoricalData.hpp"
#include "Models/MultilevelCategoricalData.hpp"
#include "distributions.hpp"

#include "test_utils/check_derivatives.hpp"
#include "test_utils/test_utils.hpp"

#include <iostream>

namespace {
  using namespace BOOM;
  using std::endl;
  using std::cout;

  // inline void print(const std::vector<std::string> &stuff) {
  //   for (const auto &el : stuff) {
  //     std::cout << el << std::endl;
  //   }
  // }

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
      input_tax_ = {
        "shopping/clothes/shoes",
        "shopping/clothes/pants",
        "shopping/food/meat",
        "shopping/food/veggies",
        "restaurants/fancy/steak",
        "restaurants/fancy/seafood",    // not a leaf!
        "restaurants/fancy/seafood/fish",
        "restaurants/fancy/seafood/lobster"
      };

      tax_.reset(new Taxonomy(input_tax_));
    }
    std::vector<std::string> input_tax_;
    Ptr<Taxonomy> tax_;
  };

  TEST_F(TaxonomyTest, input_test) {
    EXPECT_EQ(tax_->number_of_leaves(), 7);
    EXPECT_EQ(tax_->tree_size(), 13);

    // When you add a taxonomy node that is already present, nothing happens.
    std::vector<std::string> input_copy(input_tax_);
    input_copy.push_back(input_tax_.back());
    NEW(Taxonomy, tax2)(input_copy);
    EXPECT_EQ(tax2->number_of_leaves(), 7);
    EXPECT_EQ(tax2->tree_size(), 13);
  }

  TEST_F(TaxonomyTest, TestName) {
    EXPECT_EQ(tax_->name({0, 0, 0}),
              "restaurants/fancy/seafood");
    EXPECT_EQ(tax_->name({0, 0, 0, 0}),
              "restaurants/fancy/seafood/fish");
    EXPECT_EQ(tax_->name({0, 0, 0, 1}),
              "restaurants/fancy/seafood/lobster");
    EXPECT_EQ(tax_->name({0, 0, 1}),
              "restaurants/fancy/steak");
    EXPECT_EQ(tax_->name({0}),
              "restaurants");
    EXPECT_EQ(tax_->name({1}),
              "shopping");
    EXPECT_EQ(tax_->name({1, 0}),
              "shopping/clothes");
    EXPECT_EQ(tax_->name({1, 0, 0}),
              "shopping/clothes/pants");
    EXPECT_EQ(tax_->name({1, 0, 1}),
              "shopping/clothes/shoes");
    EXPECT_EQ(tax_->name({1, 1}),
              "shopping/food");
    EXPECT_EQ(tax_->name({1, 1, 0}),
              "shopping/food/meat");
    EXPECT_EQ(tax_->name({1, 1, 1}),
              "shopping/food/veggies");
  }

  void check_node(const TaxonomyNode *node, const std::string &name, int position) {
    EXPECT_EQ(node->position(), position);
    EXPECT_EQ(node->path_from_root(), name);
  }

  TEST_F(TaxonomyTest, TestNodes) {
    check_node(tax_->top_level_node(0), "restaurants", 0);
    check_node(tax_->top_level_node(0)->find_child(0),
              "restaurants/fancy", 0);
    check_node(tax_->top_level_node(0)->find_child(0)->find_child(0),
              "restaurants/fancy/seafood", 0);
    check_node(tax_->top_level_node(0)->find_child(0)->find_child(1),
              "restaurants/fancy/steak", 1);

    check_node(tax_->top_level_node(1), "shopping", 1);
    check_node(tax_->top_level_node(1)->find_child(0),
               "shopping/clothes", 0);
    check_node(tax_->top_level_node(1)->find_child(1),
               "shopping/food", 1);


    check_node(tax_->node("restaurants/fancy/seafood"),
               "restaurants/fancy/seafood", 0);
    check_node(tax_->node("shopping/food"),
               "shopping/food", 1);

    check_node(tax_->node({"restaurants", "fancy", "seafood"}),
               "restaurants/fancy/seafood", 0);
    check_node(tax_->node(std::vector<std::string>{"shopping", "food"}),
               "shopping/food", 1);
  }

  void check_index(const std::vector<int> &v1, const std::vector<int> &v2) {
    EXPECT_EQ(v1.size(), v2.size()) << "Indices are not the same size!";
    for (int i = 0; i < v1.size(); ++i) {
      EXPECT_EQ(v1[i], v2[i]) << "indices differ in position " << i;
    }
  }

  TEST_F(TaxonomyTest, Index) {
    check_index(tax_->index({"restaurants"}), {0});
    check_index(tax_->index({"restaurants", "fancy"}), {0, 0});
    check_index(tax_->index({"restaurants", "fancy", "seafood"}), {0, 0, 0});
    check_index(tax_->index({"restaurants", "fancy", "steak"}), {0, 0, 1});
    check_index(tax_->index({"restaurants", "fancy", "seafood", "fish"}),
                {0, 0, 0, 0});
    check_index(tax_->index({"restaurants", "fancy", "seafood", "lobster"}),
                {0, 0, 0, 1});

    check_index(tax_->index({"shopping"}), {1});
    check_index(tax_->index({"shopping", "clothes"}), {1, 0});
    check_index(tax_->index({"shopping", "clothes", "shoes"}), {1, 0, 1});
    check_index(tax_->index({"shopping", "clothes", "pants"}), {1, 0, 0});
    check_index(tax_->index({"shopping", "food"}), {1, 1});
    check_index(tax_->index({"shopping", "food"}), {1, 1});
    check_index(tax_->index({"shopping", "food", "meat"}), {1, 1, 0});
    check_index(tax_->index({"shopping", "food", "veggies"}), {1, 1, 1});
  }

  TEST_F(TaxonomyTest, TestIteration) {
    NEW(Taxonomy, tax)(input_tax_);
    std::vector<std::string> output;
    for (TaxonomyIterator it = tax->begin(); it != tax->end(); ++it) {
      output.push_back(it->path_from_root());
    }

    // Add the interior nodes.
    std::vector<std::string> full_tree_output = input_tax_;
    full_tree_output.push_back("shopping");
    full_tree_output.push_back("shopping/clothes");
    full_tree_output.push_back("shopping/food");
    full_tree_output.push_back("restaurants");
    full_tree_output.push_back("restaurants/fancy");

    std::sort(full_tree_output.begin(), full_tree_output.end());
    std::sort(output.begin(), output.end());

    EXPECT_EQ(full_tree_output.size(), output.size());

    NEW(Taxonomy, empty_tax)(std::vector<std::string>());
    int counter = 0;
    for (const auto &el : *empty_tax) {
      std::cout << el->path_from_root() << std::endl;
      ++counter;
    }
    EXPECT_EQ(counter, 0);
  }

  class MultilevelCategoricalDataTest : public ::testing::Test {
   protected:
    MultilevelCategoricalDataTest() {
      GlobalRng::rng.seed(8675309);
      input_tax_ = {
        "shopping/clothes/shoes",
        "shopping/clothes/pants",
        "shopping/food/meat",
        "shopping/food/veggies",
        "restaurants/fancy/steak",
        "restaurants/fancy/seafood",    // not a leaf!
        "restaurants/fancy/seafood/fish",
        "restaurants/fancy/seafood/lobster"
      };
      tax_.reset(new Taxonomy(input_tax_));
    }

    std::vector<std::string> input_tax_;
    Ptr<Taxonomy> tax_;
  };

  // Instantiating a MultilevelCategoricalData with a level not in the taxonomy
  // should throw an exception.
  TEST_F(MultilevelCategoricalDataTest, TestInvalidInput) {
    std::string level = "restaurants/fast_food";
    EXPECT_THROW(MultilevelCategoricalData data_point(tax_, level),
                 std::runtime_error);

    EXPECT_THROW(MultilevelCategoricalData dp2(tax_, "restaurants/fancy/pasta"),
                 std::runtime_error);
  }

  TEST_F(MultilevelCategoricalDataTest, TestGetSet) {

    std::string level = "restaurants/fancy/seafood/fish";

    NEW(MultilevelCategoricalData, data_point)(tax_, level);

    EXPECT_EQ(data_point->levels().size(), 4);
    EXPECT_EQ(data_point->name(), "restaurants/fancy/seafood/fish");
    std::vector<int> expected_levels = {0, 0, 0, 0};
    for (int i = 0; i < expected_levels.size(); ++i) {
      EXPECT_EQ(data_point->levels()[i], expected_levels[i]);
    }

    // Verify that a data point created using a vector of levels acts the same
    // as a data point created using a mashed together level string.
    std::vector<std::string> levels = {
      "restaurants", "fancy", "seafood", "fish"
    };
    NEW(MultilevelCategoricalData, dp2)(tax_, levels);
    EXPECT_EQ(dp2->levels().size(), 4);
    EXPECT_EQ(dp2->name(), "restaurants/fancy/seafood/fish");
    expected_levels = {0, 0, 0, 0};
    for (int i = 0; i < expected_levels.size(); ++i) {
      EXPECT_EQ(dp2->levels()[i], expected_levels[i]);
    }

    NEW(MultilevelCategoricalData, dp3)(tax_, "shopping/clothes");
    EXPECT_EQ(dp3->levels().size(), 2);
    EXPECT_EQ(dp3->name(), "shopping/clothes");
    expected_levels = {1, 0};
    for (int i = 0; i < expected_levels.size(); ++i) {
      EXPECT_EQ(dp3->levels()[i], expected_levels[i]);
    }

    NEW(MultilevelCategoricalData, dp4)(tax_, std::vector<int>{1, 0});
    EXPECT_EQ(dp4->name(), "shopping/clothes");

    NEW(MultilevelCategoricalData, dp5)(tax_, std::vector<int>{1, 0, 0});
    EXPECT_EQ(dp5->name(), "shopping/clothes/pants");

    NEW(MultilevelCategoricalData, dp6)(tax_, std::vector<int>{0, 0, 0, 1});
    EXPECT_EQ(dp6->name(), "restaurants/fancy/seafood/lobster");
  }

}  // namespace

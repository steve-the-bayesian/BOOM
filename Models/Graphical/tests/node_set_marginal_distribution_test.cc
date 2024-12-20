#include "gtest/gtest.h"

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/DummyNode.hpp"
#include "Models/Graphical/MultinomialNode.hpp"
#include "Models/Graphical/NodeSet.hpp"
#include "Models/Graphical/NodeSetMarginalDistribution.hpp"
#include "distributions.hpp"
#include "distributions/rng.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::Graphical;

  class NodeSetMarginalDistributionTest : public ::testing::Test {
   protected:
    NodeSetMarginalDistributionTest() {
      GlobalRng::rng.seed(8675309);
    }

  };

  TEST_F(NodeSetMarginalDistributionTest, smoke_test) {

  }

  TEST_F(NodeSetMarginalDistributionTest, TestMargin) {
    // Both the data and the Node's need lists of levels for the categorical
    // variables.
    NEW(CatKey, stooge_levels)(std::vector<std::string>{
        "Larry", "Moe", "Curly"});
    NEW(CatKey, color_levels)(std::vector<std::string>{
        "Red", "Blue", "Green"});
    NEW(CatKey, region_levels)(std::vector<std::string>{
        "North", "South", "East", "West"});

    NEW(MultinomialNode, Stooge)(0, "Stooge", 0, stooge_levels);
    NEW(MultinomialNode, Color)(1, "Color", 1, color_levels);
    NEW(MultinomialNode, Region)(2, "Region", 2, region_levels);

    Stooge->add_child(Color);
    Color->add_child(Region);

    NEW(LabeledCategoricalData, stooge)(1, stooge_levels);
    NEW(LabeledCategoricalData, color)(1, color_levels);
    NEW(LabeledCategoricalData, region)(2, region_levels);

    NEW(MixedMultivariateData, data_point)();
    data_point->add_categorical(stooge, "Stooge");
    data_point->add_categorical(color, "Color");
    data_point->add_categorical(region, "Region");


    NodeSet node_set;
    node_set.add(Stooge);
    node_set.add(Color);
    node_set.add(Region);

    color->set_missing_status(Data::missing_status::completely_missing);
    region->set_missing_status(Data::missing_status::completely_missing);

    NodeSetMarginalDistribution dist(&node_set);
    dist.initialize_forward(*data_point);

    Array joint_dist{3, 3, 4};
    joint_dist.randomize();
    joint_dist /= joint_dist.sum();

    Array color_region_margin = joint_dist.sum(std::vector<int>{0});
    EXPECT_EQ(color_region_margin.ndim(), 2);
    EXPECT_EQ(color_region_margin.dim(0), 3);
    EXPECT_EQ(color_region_margin.dim(1), 4);
    EXPECT_DOUBLE_EQ(color_region_margin(0, 0),
                     joint_dist(0, 0, 0)
                     + joint_dist(1, 0, 0)
                     + joint_dist(2, 0, 0));

    EXPECT_DOUBLE_EQ(color_region_margin(2, 1),
                     joint_dist(0, 2, 1)
                     + joint_dist(1, 2, 1)
                     + joint_dist(2, 2, 1));

    EXPECT_EQ(dist.unknown_discrete_distribution().ndim(), 2);
    dist.set_unknown_discrete_distribution(color_region_margin);


    NodeSetMarginalDistribution stooge_color_margin = dist.compute_margin(
        NodeSet{Stooge, Color});
    EXPECT_EQ(stooge_color_margin.unknown_discrete_nodes(), NodeSet{Color});
    std::map<Ptr<Node>, int> result;
    result[Stooge] = 1;
    EXPECT_EQ(stooge_color_margin.known_discrete_variables(), result);
    EXPECT_EQ(stooge_color_margin.unknown_discrete_distribution(),
              color_region_margin.sum(std::vector<int>{1}));

    NodeSetMarginalDistribution color_margin = dist.compute_margin(NodeSet{Color});
    EXPECT_EQ(color_margin.unknown_discrete_nodes(), NodeSet{Color});
    result.clear();
    EXPECT_EQ(color_margin.known_discrete_variables(), result);
    EXPECT_EQ(color_margin.unknown_discrete_distribution(),
              color_region_margin.sum(std::vector<int>{1}));
  }
}

#include "gtest/gtest.h"

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/DummyNode.hpp"
#include "distributions.hpp"
#include "distributions/rng.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::Graphical;

  class NodeTest : public ::testing::Test {
   protected:
    NodeTest() {
      GlobalRng::rng.seed(8675309);
    }
  };

  TEST_F(NodeTest, test_dummy_node) {
    NEW(DummyNode, fred)(1, "Fred");
    NEW(DummyNode, barney)(2, "Barney");
    NEW(DummyNode, wilma)(3, "Wilma");
    NEW(DummyNode, pebbles)(4, "Pebbles");
    NEW(DummyNode, betty)(5, "Betty");
    NEW(DummyNode, bambam)(6, "BamBam");

    EXPECT_EQ(fred->id(), 1);
    EXPECT_EQ(betty->name(), "Betty");
    EXPECT_EQ(fred->node_type(), NodeType::DUMMY);

    fred->add_child(pebbles);
    wilma->add_child(pebbles);
    EXPECT_TRUE(pebbles->is_neighbor(fred));
    EXPECT_TRUE(pebbles->is_neighbor(wilma));

    EXPECT_EQ(fred->children().size(), 1);
    EXPECT_EQ(fred->children()[0]->name(), "Pebbles");
    EXPECT_TRUE(pebbles->is_child(fred));

    EXPECT_EQ(pebbles->parents().size(), 2);
    EXPECT_EQ(pebbles->parents()[0]->name(), "Fred");

    EXPECT_TRUE(pebbles->is_child(wilma));
    EXPECT_TRUE(fred->is_parent(pebbles));
    EXPECT_TRUE(wilma->is_parent(pebbles));

    EXPECT_FALSE(pebbles->is_neighbor(pebbles));
    EXPECT_FALSE(pebbles->is_neighbor(barney));
    EXPECT_FALSE(pebbles->is_neighbor(bambam));

    bambam->add_parent(barney);
    bambam->add_parent(betty);

  }
}

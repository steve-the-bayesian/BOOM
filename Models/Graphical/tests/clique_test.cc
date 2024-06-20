#include "gtest/gtest.h"

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/DummyNode.hpp"
#include "Models/Graphical/Clique.hpp"
#include "distributions.hpp"
#include "distributions/rng.hpp"

namespace {
  using namespace BOOM;
  using namespace BOOM::Graphical;

  class CliqueTest : public ::testing::Test {
   protected:
    CliqueTest() {
      GlobalRng::rng.seed(8675309);
    }

    void MakeFlintstones() {
      NEW(DummyNode, fred)(0, "Fred");
      NEW(DummyNode, barney)(1, "Barney");
      NEW(DummyNode, wilma)(2, "Wilma");
      NEW(DummyNode, pebbles)(3, "Pebbles");
      NEW(DummyNode, betty)(4, "Betty");
      NEW(DummyNode, bambam)(5, "Bambam");
      NEW(DummyNode, dino)(6, "Dino");

      nodes_.push_back(fred);
      nodes_.push_back(barney);
      nodes_.push_back(wilma);
      nodes_.push_back(pebbles);
      nodes_.push_back(betty);
      nodes_.push_back(bambam);
      nodes_.push_back(dino);

      fred->add_child(pebbles);
      wilma->add_child(pebbles);
      barney->add_child(bambam);
      betty->add_child(bambam);
      pebbles->add_child(dino);
    }

    std::vector<Ptr<Node>> nodes_;
  };

  TEST_F(CliqueTest, test_empty_graph) {
    std::vector<Ptr<Clique>> cliques = find_cliques(nodes_);
    EXPECT_TRUE(cliques.empty());
  }

  TEST_F(CliqueTest, test_singleton_graph) {
    nodes_.push_back(new DummyNode(0, "Foo"));
    std::vector<Ptr<Clique>> cliques = find_cliques(nodes_);
    EXPECT_EQ(cliques.size(), 1);
    EXPECT_EQ(cliques[0]->elements().size(), 1);
    EXPECT_EQ(cliques[0]->elements()[0]->id(), 0);
    EXPECT_EQ(cliques[0]->elements()[0]->name(),  "Foo");
  }

  TEST_F(CliqueTest, test_flinstones) {
    MakeFlintstones();
    std::vector<Ptr<Clique>> cliques = find_cliques(nodes_);

    /*
       Fred   Wilma      Barney  Betty
          \  /               \   /
           Pebbles           Bambam
           |
           Dino

       In this graph, there are 5 cliques.
     */

    EXPECT_EQ(cliques.size(), 5);
    EXPECT_EQ(cliques[0]->elements()[0]->name(), "Fred");
    EXPECT_EQ(cliques[0]->elements()[1]->name(), "Pebbles");

    EXPECT_EQ(cliques[1]->elements()[0]->name(), "Barney");
    EXPECT_EQ(cliques[1]->elements()[1]->name(), "Bambam");

    EXPECT_EQ(cliques[2]->elements()[0]->name(), "Wilma");
    EXPECT_EQ(cliques[2]->elements()[1]->name(), "Pebbles");

    EXPECT_EQ(cliques[3]->elements()[0]->name(), "Betty");
    EXPECT_EQ(cliques[3]->elements()[1]->name(), "Bambam");

    EXPECT_EQ(cliques[4]->elements()[0]->name(), "Pebbles");
    EXPECT_EQ(cliques[4]->elements()[1]->name(), "Dino");
  }
}

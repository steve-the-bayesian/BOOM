#include "gtest/gtest.h"

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/DummyNode.hpp"
#include "Models/Graphical/Clique.hpp"
#include "Models/Graphical/JunctionTree.hpp"
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

    Ptr<MoralNode> find_moral_by_name(const std::vector<Ptr<MoralNode>> &nodes,
                                      const std::string &name) {
      for (const auto &el : nodes) {
        if (el->name() == name) {
          return el;
        }
      }
      return nullptr;
    }

    std::vector<Ptr<DummyNode>> nodes_;
  };

  std::vector<Ptr<MoralNode>> moralize(const std::vector<Ptr<DummyNode>> &nodes) {
    std::vector<Ptr<DirectedNode>> directed(nodes.begin(), nodes.end());
    return create_moral_graph(directed);
  }

  TEST_F(CliqueTest, test_empty_graph) {
    std::vector<Ptr<MoralNode>> nodes;
    UndirectedGraph<Ptr<Clique>> cliques = find_cliques(nodes);
    EXPECT_TRUE(cliques.empty());
  }

  TEST_F(CliqueTest, test_singleton_graph) {
    nodes_.push_back(new DummyNode(0, "Foo"));
    std::vector<Ptr<MoralNode>> moral_graph = moralize(nodes_);
    UndirectedGraph<Ptr<Clique>> cliques = find_cliques(moral_graph);
    EXPECT_EQ(cliques.size(), 1);
    Ptr<Clique> clique = *cliques.begin();
    EXPECT_EQ(clique->elements().size(), 1);
    EXPECT_EQ(clique->elements()[0]->id(), 0);
    EXPECT_EQ(clique->elements()[0]->name(),  "Foo");
  }

  TEST_F(CliqueTest, test_flinstones) {
    MakeFlintstones();
    std::vector<Ptr<MoralNode>> moral = moralize(nodes_);
    Ptr<MoralNode> moral_fred = find_moral_by_name(moral, "Fred");
    EXPECT_TRUE(!!moral_fred);
    EXPECT_EQ(moral_fred->neighbors().size(), 2);

    Ptr<MoralNode> moral_pebbles = find_moral_by_name(moral, "Pebbles");
    EXPECT_EQ(moral_pebbles->neighbors().size(), 3);

    UndirectedGraph<Ptr<Clique>> cliques = find_cliques(moral);

    /*
      The directed graph (arrows point down) looks like this:
       Fred   Wilma      Barney  Betty
          \  /               \   /
           Pebbles           Bambam
           |
           Dino

       In this graph, there are 5 cliques. The moral graph has 3 cliques.
     */

    EXPECT_EQ(cliques.size(), 3);
    // cliques.print(std::cout);
    // std::cout << cliques;

    // EXPECT_EQ(cliques[0]->elements()[0]->name(), "Fred");
    // EXPECT_EQ(cliques[0]->elements()[1]->name(), "Wilma");
    // EXPECT_EQ(cliques[0]->elements()[2]->name(), "Pebbles");

    // EXPECT_EQ(cliques[1]->elements()[0]->name(), "Barney");
    // EXPECT_EQ(cliques[1]->elements()[1]->name(), "Betty");
    // EXPECT_EQ(cliques[1]->elements()[2]->name(), "Bambam");

    // EXPECT_EQ(cliques[2]->elements()[0]->name(), "Pebbles");
    // EXPECT_EQ(cliques[2]->elements()[1]->name(), "Dino");
  }
}

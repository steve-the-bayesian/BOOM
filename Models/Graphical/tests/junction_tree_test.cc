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

  class JunctionTreeTest : public ::testing::Test {
   protected:
    JunctionTreeTest() {
      GlobalRng::rng.seed(8675309);
    }

    void BuildNodes(int num_nodes) {
      char name = 'A';
      for (int i = 0; i < num_nodes; ++i) {
        nodes_.push_back(new DummyNode(i, std::string(1, name++)));
      }
    }

    std::vector<Ptr<DirectedNode>> nodes_;
  };

  // TEST_F(JunctionTreeTest, TestEmptyGraph) {
  //   std::vector<Ptr<DirectedNode>> empty;
  //   JunctionTree jtree(empty);
  // }

  TEST_F(JunctionTreeTest, TestLinearGraph) {
    BuildNodes(3);
    Ptr<DirectedNode> A = nodes_[0];
    Ptr<DirectedNode> B = nodes_[1];
    Ptr<DirectedNode> C = nodes_[2];
    A->add_child(B);
    B->add_child(C);

    JunctionTree jtree(nodes_);

    EXPECT_EQ(jtree.number_of_nodes(), 3);
    EXPECT_EQ(jtree.number_of_cliques(), 2);

    jtree.build(nodes_);
  }

  // Check to see that the graphs from the Cowell et al book are correctly
  // junction-treed.
  //
  // In this graph, a C-B link should be added.
  //   A → B       A - B
  //   ↓   ↓  ---> | / |
  //   C → D       C - D
  TEST_F(JunctionTreeTest, TestSquareGraph) {
    BuildNodes(4);
    Ptr<DirectedNode> A = nodes_[0];
    Ptr<DirectedNode> B = nodes_[1];
    Ptr<DirectedNode> C = nodes_[2];
    Ptr<DirectedNode> D = nodes_[3];
    A->add_child(B);
    A->add_child(C);
    B->add_child(D);
    C->add_child(D);

    JunctionTree jtree(nodes_);
    EXPECT_EQ(jtree.number_of_nodes(), 4);
    EXPECT_EQ(jtree.number_of_cliques(), 2);
  }

  //  A   B   E         A  B   F
  //  ↓   ↓   ↓         | /| / |
  //  C → D → F ---->   C -D - E
  //
  // The cliques in this graph are
  TEST_F(JunctionTreeTest, TestFigure4_6) {
    BuildNodes(6);
    Ptr<DirectedNode> A = nodes_[0];
    Ptr<DirectedNode> B = nodes_[1];
    Ptr<DirectedNode> C = nodes_[2];
    Ptr<DirectedNode> D = nodes_[3];
    Ptr<DirectedNode> E = nodes_[4];
    Ptr<DirectedNode> F = nodes_[5];

    A->add_child(C);
    B->add_child(D);
    C->add_child(D);
    D->add_child(F);
    E->add_child(F);

    JunctionTree jtree(nodes_);
    EXPECT_EQ(jtree.number_of_nodes(), 6);
    EXPECT_EQ(jtree.number_of_cliques(), 3);
  }

  //    A                  A
  //   ↙ ↘                / \
  //  B → C   --->       B - C
  //  ↓ ↘ ↓              | \ |
  //  D ← E              D - E
  //
  // Note that the moral graph is not triangular, because B-D-E-C-B is a cycle,
  // but there is no link between C and D.  The triangularization algorithm
  // introduces this link, so there are two cliques in the triangulated moral
  // graph: ABC, and BCDE.
  TEST_F(JunctionTreeTest, Figure4_3) {
    BuildNodes(5);
    Ptr<DirectedNode> A = nodes_[0];
    Ptr<DirectedNode> B = nodes_[1];
    Ptr<DirectedNode> C = nodes_[2];
    Ptr<DirectedNode> D = nodes_[3];
    Ptr<DirectedNode> E = nodes_[4];

    A->add_child(B);
    A->add_child(C);
    B->add_child(C);
    B->add_child(D);
    B->add_child(E);
    C->add_child(E);
    E->add_child(D);

    JunctionTree jtree(nodes_);
    EXPECT_EQ(jtree.number_of_nodes(), 5);
    EXPECT_EQ(jtree.number_of_cliques(), 2);
  }

}

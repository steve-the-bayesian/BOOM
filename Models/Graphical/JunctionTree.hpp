#ifndef BOOM_MODELS_GRAPHICAL_JUNCTION_TREE_HPP_
#define BOOM_MODELS_GRAPHICAL_JUNCTION_TREE_HPP_

/*
  Copyright (C) 2005-2024 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/NodeSet.hpp"
#include "Models/Graphical/Clique.hpp"
#include "Models/Graphical/NodeSetMarginalDistribution.hpp"
#include "Models/Graphical/UndirectedGraph.hpp"

#include <functional>
#include <vector>
#include <set>
#include "cpputil/SortedVector.hpp"

namespace BOOM {

  namespace Graphical {

    // A functor used as part of the graph triangulation algoirithm.  Produces a
    // "cost" of selecting a given node to focus on during triangulation.
    class DefaultTriangulationHeuristic {
     public:
      DefaultTriangulationHeuristic(const std::vector<Ptr<Node>> &nodes);
      double operator()(const Ptr<Node> &node);
      void reset_base_nodes(const std::vector<Ptr<Node>> &nodes);

     private:
      std::vector<Ptr<Node>> moral_nodes_;
    };

    // Produce a collection of Nodes by marrying the parents of each
    // directed node, and then dropping the arrows.
    void create_moral_graph(std::vector<Ptr<Node>> &directed_nodes);

    //===========================================================================
    // A tree of cliques in the moral, triangulated graph generated from a DAG.
    class JunctionTree {
     public:
      using Criterion = std::function<double(const Ptr<Node> &)>;
      using Marginals = std::map<Ptr<Clique>,
                                 Ptr<NodeSetMarginalDistribution>>;
      using CliqueTree = UndirectedGraph<Ptr<Clique>>;
      using EliminationTree = UndirectedGraph<Ptr<NodeSet>>;

      // An empty junction tree.  The tree can be populated by a call to
      // build().
      JunctionTree();

      // A junction tree created from the set of directed nodes.
      JunctionTree(const std::vector<Ptr<Node>> &nodes);

      // Build the JunctionTree based on the supplied set of nodes.
      void build(const std::vector<Ptr<Node>> &nodes);

      // The triangulation heuristic is a function (of Nodes) that returns
      // a real number describing the "cost" of selecting that node.  It is used
      // (as a heuristic) when choosing edges to add during the triangulation
      // algorithm.
      void set_triangulation_heuristic(const Criterion &heuristic) {
        triangulation_heuristic_ = heuristic;
      }

      // Fill any missing values in data_point with values imputed from their
      // joint posterior distribution.
      void impute_missing_values(MixedMultivariateData &data_point, RNG &rng);

      // Fill the "marginals_" structure with marginal distributions describing
      // the conditional distribution of each clique given any observed data
      // seen by its ancestors.
      //
      // Returns:
      double accumulate_evidence(const MixedMultivariateData &data_point);

      const Marginals & marginals() const {
        return marginals_;
      }

      // Update the marginals_ structure so that each marginal clique
      // distribution conditions on all observed data in 'data_point'.
      void distribute_evidence(const Ptr<MixedMultivariateData> &data_point);

      size_t number_of_cliques() const {
        return clique_graph_.size();
      }

      size_t number_of_nodes() const {
        return directed_nodes_.size();
      }

      // Produce a human readable string describing the set of Cliques in the
      // junction tree.
      //
      // The string has the form
      // A:B:C | B:C:D C:E B:F
      // ...
      //
      // where the listing before the vertical bar is the name of the Clique
      // (formed by concatenating the names of its elements), and the listings
      // after the vertical bars are its neighbors.
      std::string print_cliques() const;

      // Find a root node in the tree of cliques, where at least one node has a
      // base element with no parents, and where all nodes with parents have
      // those parents present in the clique.
      //
      // Args:
      //   cliques:  A collection of cliques arranged in a tree structure.
      //
      // Returns:
      //   A legal root node in the tree.
      Ptr<Clique> find_root(const UndirectedGraph<Ptr<Clique>> &graph) const;

      // The neighbors of a given clique in the JunctionTree.
      const std::set<Ptr<Clique>> &neighbors(const Ptr<Clique> &clique) const {
        return clique_graph_.neighbors(clique);
      }

     private:
      //========================================================================
      // Implementation details for the forward-backward algorithm.
      //
      // Args:
      //   data_point:  The data point being evaluated.
      //   root:  The clique in the junction tree serving as the root node.
      //
      // Returns:
      //   The contribution to log likelihood from the information in 'root'.
      double initialize_forward_algorithm(
          const MixedMultivariateData &data_point,
          const Ptr<Clique> &root);

      // One step in the forward recursion.
      //
      // Args:
      //   data_point: The data point providing the evidence.
      //   neighbor: The clique in the junction tree around which evidence is to
      //     be collected.
      //   source: The clique in the junction tree passing the message to
      //     'clique'.
      //
      // Returns:
      //   The incremental contribution to log likelihood.
      double collect_additional_evidence(
          const MixedMultivariateData &data_point,
          const Ptr<Clique> &clique,
          const Ptr<Clique> &source);

      //========================================================================
      // Utilities used in building the junction tree

      // // Produce a collection of Nodes by marrying the parents of each
      // // directed node, and then dropping the arrows.
      // std::vector<Ptr<Node>> create_moral_graph(
      //     const std::vector<Ptr<Node>> & directed_nodes);

      // Choose the node from among &nodes that minimizes 'heuristic'.
      Ptr<Node> choose_node(
          SortedVector<Ptr<Node>> &nodes,
          Criterion &heuristic);

      // Take an undirected graph of Nodes, potentially re-order them, and
      // add edges so that the graph is triangulated.  Return the collections of
      // nodes known as Elimination Sets (according to algorithm 4.13 in Cowell
      // et al.).
      std::vector<Ptr<NodeSet>> triangulate_moral_graph(
          std::vector<Ptr<Node>> &nodes);

      // Impose a tree stucture on the collection of elimination sets.
      EliminationTree make_elimination_tree(
          std::vector<Ptr<NodeSet>> &elimination_sets);

      void prune_elimination_tree(
          std::vector<Ptr<NodeSet>> &elimination_sets,
          EliminationTree &tree,
          int start_from = 0);

      std::vector<Ptr<Clique>> make_junction_tree(
          std::vector<Ptr<NodeSet>> &elimination_sets,
          EliminationTree &tree);

      void make_dense(Ptr<NodeSet> &nodes);
      int find_second_largest_index(const NodeSet &nodes);

      //----------------------------------------------------------------------
      // Data members
      std::vector<Ptr<Node>> directed_nodes_;
      Criterion triangulation_heuristic_;

      UndirectedGraph<Ptr<Clique>> clique_graph_;
      Ptr<Clique> root_;

      // Marginals is used during calls to "accumulate_evidence" or
      // "distribute_evidence".
      Marginals marginals_;
    };

  }  // namespace Graphical

}  // namespace BOOM


#endif  //  BOOM_MODELS_GRAPHICAL_JUNCTION_TREE_HPP_

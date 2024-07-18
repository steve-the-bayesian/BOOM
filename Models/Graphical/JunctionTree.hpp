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
      DefaultTriangulationHeuristic(const std::vector<Ptr<MoralNode>> &nodes);
      double operator()(const Ptr<MoralNode> &node);
      void reset_base_nodes(const std::vector<Ptr<MoralNode>> &nodes);

     private:
      std::vector<Ptr<MoralNode>> moral_nodes_;
    };

    //===========================================================================
    // A tree of cliques in the moral, triangulated graph generated from a DAG.
    class JunctionTree {
     public:
      using Criterion = std::function<double(const Ptr<MoralNode> &)>;
      using Evidence = std::map<Ptr<Clique<MoralNode>>,
                                CliqueMarginalDistribution>;

      // An empty junction tree.  The tree can be populated by a call to
      // build().
      JunctionTree();

      // A junction tree created from the set of directed nodes.
      JunctionTree(const std::vector<Ptr<DirectedNode>> &nodes);

      // Build the JunctionTree based on the supplied set of nodes.
      void build(const std::vector<Ptr<DirectedNode>> &nodes);

      // The triangulation heuristic is a function (of MoralNodes) that returns
      // a real number describing the "cost" of selecting that node.  It is used
      // (as a heuristic) when choosing edges to add during the triangulation
      // algorithm.
      void set_triangulation_heuristic(const Criterion &heuristic) {
        triangulation_heuristic_ = heuristic;
      }

      // Fill any missing values in data_point with values imputed from their
      // joint posterior distribution.
      void impute_missing_values(MixedMultivariateData &data_point, RNG &rng);

      // Fill the "evidence_" structure with marginal distributions describing
      // the conditional distribution of each clique given any observed data
      // seen by its ancestors.
      void accumulate_evidence(const Ptr<MixedMultivariateData> &data_point);

      // Update the evidence_ structure so that each marginal clique
      // distribution conditions on all observed data in 'data_point'.
      void distribute_evidence(const Ptr<MixedMultivariateData> &data_point,
                               Evidence &evidence);

      size_t number_of_cliques() const {
        return cliques_.size();
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

     private:
      //===========================================================================
      // Utilities used in building the junction tree

      // Produce a collection of MoralNodes by marrying the parents of each
      // directed node, and then dropping the arrows.
      std::vector<Ptr<MoralNode>> create_moral_graph(
          const std::vector<Ptr<DirectedNode>> & directed_nodes);

      // Choose the node from among &nodes that minimizes 'heuristic'.
      Ptr<MoralNode> choose_node(
          SortedVector<Ptr<MoralNode>> &nodes, Criterion &heuristic);

      // Take an undirected graph of MoralNodes, potentially re-order them, and
      // add edges so that the graph is triangulated.  Return the collections of
      // nodes known as Elimination Sets (according to algorithm 4.13 in Cowell
      // et al.).
      std::vector<Ptr<NodeSet<MoralNode>>> triangulate_moral_graph(
          std::vector<Ptr<MoralNode>> &nodes);

      // Impose a tree stucture on the collection of elimination sets.
      void make_elimination_tree(
          std::vector<Ptr<NodeSet<MoralNode>>> &elimination_sets);

      void prune_elimination_tree(
          std::vector<Ptr<NodeSet<MoralNode>>> &elimination_sets,
          int start_from = 0);

      std::vector<Ptr<Clique<MoralNode>>>
      make_junction_tree_from_elimination_sets(
          std::vector<Ptr<NodeSet<MoralNode>>> &elimination_sets);

      void make_dense(Ptr<NodeSet<MoralNode>> &nodes);
      int find_second_largest_index(const NodeSet<MoralNode> &nodes);

      //----------------------------------------------------------------------
      // Data members
      std::vector<Ptr<DirectedNode>> directed_nodes_;
      Criterion triangulation_heuristic_;
      std::vector<Ptr<Clique<MoralNode>>> cliques_;

      // Evidence is used during calls to "accumulate_evidence" or
      // "distribute_evidence".
      Evidence evidence_;
    };

  }  // namespace Graphical

}  // namespace BOOM


#endif  //  BOOM_MODELS_GRAPHICAL_JUNCTION_TREE_HPP_

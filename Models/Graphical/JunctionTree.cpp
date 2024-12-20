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

#include "Models/Graphical/JunctionTree.hpp"
#include <sstream>

namespace BOOM {
  namespace Graphical {

    //---------------------------------------------------------------------------
    // Take a set of directed nodes, marry the parents of each child node, and
    // drop the arrows.  The output of this function is a collection of
    // Nodes, each of which holds a pointer to a directed base node.
    //
    // Args:
    //   directed_nodes:  Nodes of the directed graph to be moralized.
    //
    // Returns:
    //   The nodes of a moral graph, obtained by "marrying the parents" of each
    //   node in 'nodes', then dropping the arrows.
    void create_moral_graph(std::vector<Ptr<Node>> &directed_nodes) {
      for (auto &el : directed_nodes) {
        el->clear_neighbors();
      }

      for (const Ptr<Node> &node : directed_nodes) {
        // Each node is neighbors with its parents.
        for (const Ptr<Node> &parent : node->parents()) {
          node->add_neighbor(parent);
          // All of the parents are neighbors with each other.
          for (const Ptr<Node> &other_parent : node->parents()) {
            if (parent != other_parent) {
              parent->add_neighbor(other_parent);
            }
          }
        }

        // A node is neighbors with its children.
        for (const Ptr<Node> &child : node->children()) {
          node->add_neighbor(child);
        }
      }
    }


    namespace {
      // Returns true iff all parents of each node in the clique are also in the
      // clique.
      bool is_root(const Ptr<Clique> &clique) {
        for (const Ptr<Node> &el : clique->elements()) {
          for (const auto &parent : el->parents()) {
            if (!clique->contains(parent)) {
              return false;
            }
          }
        }
        return true;
      }
    }  // namespace

    std::string print_moral_node(const Ptr<Node> &node) {
      std::ostringstream out;
      out << node->id() << ' ' << node->name() << " |";
      for (const auto &neighbor : node->neighbors()) {
        out << ' ' << neighbor->id() << ' ' << neighbor->name();
      }
      return out.str();
    }

    std::string print_moral_nodes(const std::vector<Ptr<Node>> &nodes) {
      std::ostringstream out;
      out << "---- Moral Nodes: -----\n";
      for (const auto &node : nodes) {
        out << print_moral_node(node) << std::endl;
      }
      return out.str();
    }

    template <class NODESET>
    std::string print_node_set(
        const Ptr<NODESET> &set,
        const UndirectedGraph<Ptr<NODESET>> &graph) {
      std::ostringstream out;
      out << set->id() << ' ' << set->name() << " |";
      for (const auto &neighbor : graph.neighbors(set)) {
        out << ' ' << neighbor->id() << ' ' << neighbor->name();
      }
      return out.str();
    }

    template <class NODESET>
    std::string print_node_sets(const UndirectedGraph<Ptr<NODESET>> &graph) {
      std::ostringstream out;
      out << "----- Elimination Sets: -----\n";
      for (const auto &set : graph) {
        out << print_node_set(set, graph) << std::endl;
      }
      return out.str();
    }

    JunctionTree::JunctionTree()
        : triangulation_heuristic_(
              [](const Ptr<Node> &node) {
                return node->parents().size();
              })
    {}

    JunctionTree::JunctionTree(const std::vector<Ptr<Node>> &nodes)
        : JunctionTree()
    {
      build(nodes);
    }

    //---------------------------------------------------------------------------
    // This is the main function that builds the junction tree.
    //
    // Args:
    //   nodes: The nodes of a directed graph to be described by the junction
    //     tree.
    //
    // Effects:
    //   The internals of the tree are built (or rebuilt).
    void JunctionTree::build(const std::vector<Ptr<Node>> &nodes) {
      directed_nodes_ = nodes;
      create_moral_graph(directed_nodes_);
      std::vector<Ptr<NodeSet>> elimination_sets =
          triangulate_moral_graph(directed_nodes_);
      EliminationTree elimination_tree = make_elimination_tree(elimination_sets);
      prune_elimination_tree(elimination_sets, elimination_tree);
      make_junction_tree(elimination_sets, elimination_tree);
      root_ = find_root(clique_graph_);
    }

    Ptr<Clique> JunctionTree::find_root(
        const UndirectedGraph<Ptr<Clique>> &graph) const {
      if (!graph.empty()) {
        for (const Ptr<Clique> &clique: graph) {
          if (is_root(clique)) {
            return clique;
          }
        }
        report_error("Could not identify a root node.");
      }
      return nullptr;
    }

    //---------------------------------------------------------------------------
    // Accumulating evidence.  Need to start from a node without parents.
    double JunctionTree::accumulate_evidence(const MixedMultivariateData &dp) {
      double logp = initialize_forward_algorithm(dp, root_);
      for (Ptr<Clique> neighbor : neighbors(root_)) {
        logp += collect_additional_evidence(dp, neighbor, root_);
      }
      return logp;
    }

    // Args:
    //   data_point: The data point (e.g. a row in a data frame) that provides
    //     "evidence" to propagate across the junction tree.
    //   root: A root node of the junction tree.  Each node in the root must
    //     have all its parents in the root.  At least one node must have no
    //     parents.
    double JunctionTree::initialize_forward_algorithm(
        const MixedMultivariateData &data_point,
        const Ptr<Clique> &root) {
      double logp = 0;

      Ptr<NodeSetMarginalDistribution> marg = marginals_[root];
      if (!marg) {
        marg.reset(new NodeSetMarginalDistribution(root.get()));
        marginals_[root] = marg;
      }
      marg->resize(data_point);

      // Identify which nodes have no parents.
      return logp;
   }
    //---------------------------------------------------------------------------
    double JunctionTree::collect_additional_evidence(
        const MixedMultivariateData &data_point,
        const Ptr<Clique> &target,
        const Ptr<Clique> &source) {
      double logp = 0.0;
      report_error("JunctionTree::collect_additional_evidence is not yet "
                   "implemented.");

      Ptr<NodeSetMarginalDistribution> marg = marginals_[target];
      if (!marg) {
        marg.reset(new NodeSetMarginalDistribution(target.get()));
        marginals_[target] = marg;
      }

      return logp;
    }

    //---------------------------------------------------------------------------
    std::string JunctionTree::print_cliques() const {
      std::ostringstream out;
      out << "----- Cliques: -----\n";
      for (const auto &clique : clique_graph_) {
        out << print_node_set<Clique>(clique, clique_graph_) << std::endl;
      }
      return out.str();
    }

    //---------------------------------------------------------------------------
    // Args:
    //   elimination_sets: A collection of sets of Nodes produced by the
    //     triangulation algorithm.
    //
    // Effects:
    //   The elimination sets are combined into Cliques and stored in the
    //   Junction_tree

    // Implementation detail for 'ensure_junction_tree'.  Algorithm 4.8 from
    // Cowell et al.
    std::vector<Ptr<Clique>>
    JunctionTree::make_junction_tree(
        std::vector<Ptr<NodeSet>> &elimination_sets,
        EliminationTree &neighbors) {
      // Step 1: promote the elimination sets to cliques.
      std::vector<Ptr<Clique>> cliques;
      for (int i = 0; i < elimination_sets.size(); ++i) {
        cliques.push_back(new Clique(*elimination_sets[i]));
      }

      // Step 2: links between the cliques using Algorithm 4.8 from Cowell et
      // al.
      SortedVector<Ptr<Node>> running_union;
      running_union.absorb(cliques[0]->elements());
      for (int i = 1; i < cliques.size(); ++i) {
        SortedVector<Ptr<Node>> intersection =
            cliques[i]->elements().intersection(running_union);
        for (int j = 0; j < i; ++j) {
          if (!cliques[j]->elements().disjoint_from(intersection)) {
            clique_graph_.add_neighbor(cliques[i], cliques[j]);
            ////// TODO: need to copy neighbor relationships from the
            ////// elimination tree to the clique tree.
          }
        }
        running_union.absorb(cliques[i]->elements());
      }
      return cliques;
    }

    //---------------------------------------------------------------------------
    // Choose one node from nodes to optimize some function to be named later.
    // Return the chosen node, and remove it from 'nodes'.
    //
    // Part of the implementation for 'triangulate_moral_graph'.
    Ptr<Node> JunctionTree::choose_node(
        SortedVector<Ptr<Node>> &nodes,
        JunctionTree::Criterion &criterion) {
      Ptr<Node> ans = nullptr;
      double value = infinity();
      for (size_t i = 0; i < nodes.size(); ++i) {
        double candidiate_value = criterion(nodes[i]);
        if (candidiate_value < value) {
          value = candidiate_value;
          ans = nodes[i];
        }
      }
      nodes.remove(ans);
      return ans;
    }

    //---------------------------------------------------------------------------
    // Take an undirected graph of Nodes, potentially re-order them, and
    // add edges so that the graph is triangulated.  This is algorithm 4.13 from
    // Cowell et al.
    //
    // Args:
    //   nodes:  The nodes of the moral graph.
    //
    // Effects:
    //   The input graph is modified by adding neighbor relationships.
    //
    // Returns:
    //   The "Elimination Sets" from the triangulation algorithm.  See Cowell et
    //   al. (1999) section 4.4.1-4.4.2.  Element i in the return value is the set
    //   C[i] from Cowell et al algorithm 4.13.
    std::vector<Ptr<NodeSet>>
    JunctionTree::triangulate_moral_graph(std::vector<Ptr<Node>> &nodes) {
      if (nodes.empty()) {
        return std::vector<Ptr<NodeSet>>();
      }

      SortedVector<Ptr<Node>> unnumbered_nodes;
      for (auto &el : nodes) {
        el->set_id(-1);
        unnumbered_nodes.insert(el);
      }

      // The elimination sets each start off empty.
      std::vector<Ptr<NodeSet>> elimination_sets;
      for (int i = 0; i <nodes.size(); ++i) {
        elimination_sets.push_back(new NodeSet);
      }

      // Note that nodes.size() is at least one, because we bail on entry if nodes
      // is empty.
      for (int i = nodes.size() - 1; i >= 0; --i) {
        Ptr<Node> node = choose_node(unnumbered_nodes,
                                          triangulation_heuristic_);
        node->set_id(i);
        elimination_sets[i]->insert(node);
        for (const Ptr<Node> &neighbor : node->neighbors()) {
          Ptr<Node> directed_neighbor = neighbor.dcast<Node>();
          if (unnumbered_nodes.contains(directed_neighbor)) {
            elimination_sets[i]->insert(directed_neighbor);
          }
        }
        make_dense(elimination_sets[i]);
      }
      return elimination_sets;
    }

    // Algorithm 4.14 in Cowell et al.  Create a tree from the collection of
    // elimination sets produced by Algorithm 4.13.
    //
    // Args:
    //   elimination_sets: The labelled group of sets produced by algorithm 4.13
    //     (the one step lookahead triangulation algorithm).
    //
    // Effects:
    //   The elimination sets are arranged into an (undirected) elimination tree
    //   by adding neighbor links between the sets.
    JunctionTree::EliminationTree JunctionTree::make_elimination_tree(
        std::vector<Ptr<NodeSet>> &elimination_sets) {

      EliminationTree tree;
      for (size_t i = 0; i < elimination_sets.size(); ++i) {
        tree.add_element(elimination_sets[i]);
      }

      for (size_t i = 0; i < elimination_sets.size(); ++i) {
        elimination_sets[i]->set_id(i);
        if (elimination_sets[i]->size() > 1) {
          Ptr<NodeSet> neighbor = elimination_sets[
              find_second_largest_index(*elimination_sets[i])];
          tree.add_neighbor(elimination_sets[i], neighbor);
        }
      }
      return tree;
    }

    // Prune the nodes of the elimination tree using Lemma 4.16 of Cowell et al.
    //
    // For each node i, determine if the node is a proper subset of a subsequent
    // node j.  If so, bring node j forward to spot i.
    //
    // On exit, elimination_sets
    void JunctionTree::prune_elimination_tree(
        std::vector<Ptr<NodeSet>> &elimination_sets,
        JunctionTree::EliminationTree &tree,
        int start_from) {
      for (Int i = start_from; i < elimination_sets.size(); ++i) {
        for (Int j = i + 1; j < elimination_sets.size(); ++j) {
          if (elimination_sets[i]->is_subset(*elimination_sets[j])) {
            Ptr<NodeSet> redundant = elimination_sets[i];
            tree.erase(tree.find(elimination_sets[i]));
            elimination_sets[i] = elimination_sets[j];
            elimination_sets.erase(elimination_sets.begin() + j);
            prune_elimination_tree(elimination_sets, tree, i);
            return;
          }
        }
      }
    }

    //---------------------------------------------------------------------------
    // Args:
    //   nodes: An "elimination set" of moral nodes that has been labelled by the
    //     triangulation algorithm.
    // Returns:
    //   The index of the second largest node in the set (the largest index is
    //   nodes.id()).
    int JunctionTree::find_second_largest_index(
        const NodeSet &nodes) {
      int set_index = nodes.id();
      int max_index = -1;
      for (const auto &el : nodes) {
        if ((el->id() < set_index) && (el->id() > max_index)) {
          max_index = el->id();
        }
      }
      return max_index;
    }

    //---------------------------------------------------------------------------
    // Make every element of 'nodes' a neighbor of every other element.
    void JunctionTree::make_dense(Ptr<NodeSet> &nodes) {
      for (auto it = nodes->begin(); it != nodes->end(); ++it) {
        const Ptr<Node> &node = *it;
        auto other_it = it;
        ++other_it;
        for (; other_it != nodes->end(); ++other_it) {
          node->add_neighbor(*other_it);
        }
      }
    }

    void JunctionTree::impute_missing_values(
        MixedMultivariateData &data_point,
        RNG &rng) {

      report_error("Not yet implemented");

    }

  }  // namespace Graphical

}  // namespace BOOM

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

    std::string print_moral_node(const Ptr<MoralNode> &node) {
      std::ostringstream out;
      out << node->id() << ' ' << node->name() << " |";
      for (const auto &neighbor : node->neighbors()) {
        out << ' ' << neighbor->id() << ' ' << neighbor->name();
      }
      return out.str();
    }

    std::string print_moral_nodes(const std::vector<Ptr<MoralNode>> &nodes) {
      std::ostringstream out;
      out << "---- Moral Nodes: -----\n";
      for (const auto &node : nodes) {
        out << print_moral_node(node) << std::endl;
      }
      return out.str();
    }

    std::string print_node_set(const Ptr<NodeSet<MoralNode>> &elimination_set) {
      std::ostringstream out;
      out << elimination_set->id() << ' ' << elimination_set->name() << " |";
      for (const auto &neighbor : elimination_set->neighbors()) {
        out << ' ' << neighbor->id() << ' ' << neighbor->name();
      }
      return out.str();
    }

    std::string print_node_sets(
        const std::vector<Ptr<NodeSet<MoralNode>>> &sets) {
      std::ostringstream out;
      out << "----- Elimination Sets: -----\n";
      for (const auto &elimination_set : sets) {
        out << print_node_set(elimination_set) << std::endl;
      }
      return out.str();
    }

    JunctionTree::JunctionTree()
        : triangulation_heuristic_(
              [](const Ptr<MoralNode> &node) {
                return node->base_node()->parents().size();
              })
    {}

    JunctionTree::JunctionTree(const std::vector<Ptr<DirectedNode>> &nodes)
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
    void JunctionTree::build(const std::vector<Ptr<DirectedNode>> &nodes) {
      directed_nodes_ = nodes;
      std::vector<Ptr<MoralNode>> moral_nodes =
          create_moral_graph(directed_nodes_);
      std::vector<Ptr<NodeSet<MoralNode>>> elimination_sets =
          triangulate_moral_graph(moral_nodes);
      make_elimination_tree(elimination_sets);
      prune_elimination_tree(elimination_sets);
      cliques_ = make_junction_tree_from_elimination_sets(
          elimination_sets);
    }

    //---------------------------------------------------------------------------
    std::string JunctionTree::print_cliques() const {
      std::ostringstream out;
      out << "----- Cliques: -----\n";
      for (const auto &clique : cliques_) {
        out << print_node_set(clique) << std::endl;
      }
      return out.str();
    }

    //---------------------------------------------------------------------------
    // Take a set of directed nodes, marry the parents of each child node, and
    // drop the arrows.  The output of this function is a collection of
    // MoralNodes, each of which holds a pointer to a directed base node.
    //
    // Args:
    //   directed_nodes:  Nodes of the directed graph to be moralized.
    //
    // Returns:
    //   The nodes of a moral graph, obtained by "marrying the parents" of each
    //   node in 'nodes', then dropping the arrows.
    std::vector<Ptr<MoralNode>> JunctionTree::create_moral_graph(
        const std::vector<Ptr<DirectedNode>> &directed_nodes) {

      std::map<Ptr<DirectedNode>, Ptr<MoralNode>, IdLess> moral_nodes;
      for (const Ptr<DirectedNode> &base_node : directed_nodes) {
        NEW(MoralNode, moral_node)(base_node);
        moral_nodes[base_node] = moral_node;
      }

      for (const Ptr<DirectedNode> &base_node : directed_nodes) {
        Ptr<MoralNode> moral_node = moral_nodes[base_node];
        // Each node is neighbors with its parents.
        for (const Ptr<DirectedNode> &node : base_node->parents()) {
          moral_node->add_neighbor(moral_nodes[node]);
          // All of the parents are neighbors with each other.
          for (const Ptr<DirectedNode> &other_node : base_node->parents()) {
            if (node != other_node) {
              moral_nodes[node]->add_neighbor(moral_nodes[other_node]);
            }
          }
        }

        // A node is neighbors with its children.
        for (const Ptr<DirectedNode> &node : base_node->children()) {
          moral_node->add_neighbor(moral_nodes[node]);
        }
      }
      std::vector<Ptr<MoralNode>> ans;
      for (const auto &el : moral_nodes) {
        ans.push_back(el.second);
      }
      return ans;
    }
    //---------------------------------------------------------------------------
    // Args:
    //   elimination_sets: A collection of sets of MoralNodes produced by the
    //     triangulation algorithm.
    //
    // Effects:
    //   The elimination sets are combined into Cliques and stored in the
    //   Junction_tree

    // Implementation detail for 'ensure_junction_tree'.  Algorithm 4.8 from
    // Cowell et al.
    std::vector<Ptr<Clique<MoralNode>>>
    JunctionTree::make_junction_tree_from_elimination_sets(
        std::vector<Ptr<NodeSet<MoralNode>>> &elimination_sets) {
      // Step 1: promote the elimination sets to cliques.
      std::vector<Ptr<Clique<MoralNode>>> cliques;
      for (int i = 0; i < elimination_sets.size(); ++i) {
        cliques.push_back(new Clique<MoralNode>(*elimination_sets[i]));
      }

      // Step 2: links between the cliques using Algorithm 4.8 from Cowell et
      // al.
      SortedVector<Ptr<MoralNode>> running_union;
      running_union.absorb(cliques[0]->elements());
      for (int i = 1; i < cliques.size(); ++i) {
        SortedVector<Ptr<MoralNode>> intersection =
            cliques[i]->elements().intersection(running_union);
        for (int j = 0; j < i; ++j) {
          if (!cliques[j]->elements().disjoint_from(intersection)) {
            cliques[i]->add_neighbor(cliques[j]);
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
    Ptr<MoralNode> JunctionTree::choose_node(
        SortedVector<Ptr<MoralNode>> &nodes,
        JunctionTree::Criterion &criterion) {
      Ptr<MoralNode> ans = nullptr;
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
    // Take an undirected graph of MoralNodes, potentially re-order them, and
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
    std::vector<Ptr<NodeSet<MoralNode>>>
    JunctionTree::triangulate_moral_graph(std::vector<Ptr<MoralNode>> &nodes) {
      if (nodes.empty()) {
        return std::vector<Ptr<NodeSet<MoralNode>>>();
      }

      SortedVector<Ptr<MoralNode>> unnumbered_nodes;
      for (auto &el : nodes) {
        el->set_id(-1);
        unnumbered_nodes.insert(el);
      }

      // The elimination sets each start off empty.
      std::vector<Ptr<NodeSet<MoralNode>>> elimination_sets;
      for (int i = 0; i <nodes.size(); ++i) {
        elimination_sets.push_back(new NodeSet<MoralNode>);
      }

      // Note that nodes.size() is at least one, because we bail on entry if nodes
      // is empty.
      for (int i = nodes.size() - 1; i >= 0; --i) {
        Ptr<MoralNode> node = choose_node(unnumbered_nodes,
                                          triangulation_heuristic_);
        node->set_id(i);
        elimination_sets[i]->insert(node);
        for (const Ptr<MoralNode> &neighbor : node->neighbors()) {
          if (unnumbered_nodes.contains(neighbor)) {
            elimination_sets[i]->insert(neighbor);
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
    void JunctionTree::make_elimination_tree(
        std::vector<Ptr<NodeSet<MoralNode>>> &elimination_sets) {
      for (size_t i = 0; i < elimination_sets.size(); ++i) {
        elimination_sets[i]->set_id(i);
        if (elimination_sets[i]->size() > 1) {
          elimination_sets[i]->add_neighbor(
              elimination_sets[find_second_largest_index(
                  *elimination_sets[i])]);
        }
      }
    }

    // Prune the nodes of the elimination tree using Lemma 4.16 of Cowell et al.
    //
    // For each node i, determine if the node is a proper subset of a subsequent
    // node j.  If so, bring node j forward to spot i.
    //
    // On exit, elimination_sets
    void JunctionTree::prune_elimination_tree(
        std::vector<Ptr<NodeSet<MoralNode>>> &elimination_sets,
        int start_from) {
      for (Int i = start_from; i < elimination_sets.size(); ++i) {
        for (Int j = i + 1; j < elimination_sets.size(); ++j) {
          if (elimination_sets[i]->is_subset(*elimination_sets[j])) {
            elimination_sets[i] = elimination_sets[j];
            elimination_sets.erase(elimination_sets.begin() + j);
            prune_elimination_tree(elimination_sets, i);
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
        const NodeSet<MoralNode> &nodes) {
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
    void JunctionTree::make_dense(Ptr<NodeSet<MoralNode>> &nodes) {
      for (auto it = nodes->begin(); it != nodes->end(); ++it) {
        const Ptr<MoralNode> &node = *it;
        auto other_it = it;
        ++other_it;
        for (; other_it != nodes->end(); ++other_it) {
          node->add_neighbor(*other_it);
        }
      }
    }

  }  // namespace Graphical

}  // namespace BOOM

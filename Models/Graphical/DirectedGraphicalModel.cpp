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

#include "Models/Graphical/DirectedGraphicalModel.hpp"
#include "cpputil/report_error.hpp"
#include <map>
#include <functional>

namespace BOOM {
  using namespace Graphical;

  DirectedGraphicalModel::DirectedGraphicalModel()
      : junction_tree_current_(false)
  {}

  void DirectedGraphicalModel::add_node(const Ptr<DirectedNode> &node) {
    if (nodes_.contains(node)) {
      std::ostringstream err;
      err << "A node with id " << node->id() << " and name "
          << node->name() << " already exists in the model.";
      report_error(err.str());
    }

    nodes_.insert(node);
    junction_tree_current_ = false;
  }

  void DirectedGraphicalModel::accumulate_evidence(
      const Ptr<MixedMultivariateData> &dp) {
    ensure_junction_tree();

  }

  //===========================================================================
  // Take a set of directed nodes, marry the parents of each child node, and
  // drop the arrows.  The output of this function is a collection of
  // MoralNodes, each of which holds a pointer to a directed base node.
  //
  // Args:
  //   directed_nodes:  Nodes of the directed graph to be moralized.
  std::vector<Ptr<MoralNode>>
  DirectedGraphicalModel::create_moral_graph(
      const std::vector<Ptr<DirectedNode>> &directed_nodes) const {

    std::map<Ptr<DirectedNode>, Ptr<MoralNode>> moral_nodes;
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

  //===========================================================================
  // Choose one node from nodes to optimize some function to be named later.
  // Return the chosen node, and remove it from 'nodes'.
  Ptr<MoralNode> choose_node(
      SortedVector<Ptr<MoralNode>> &nodes,
      const std::function<double(const Ptr<MoralNode>)> & criterion) {
    Ptr<MoralNode> ans = nullptr;
    double value = negative_infinity();
    for (size_t i = 0; i < nodes.size(); ++i) {
      double candidiate_value = criterion(nodes[i]);
      if (candidiate_value > value) {
        value = candidiate_value;
        ans = nodes[i];
      }
    }
    nodes.remove(ans);
    return ans;
  }

  //===========================================================================
  // Make every element of 'nodes' a neighbor of every other element.
  void make_dense(const std::set<Ptr<MoralNode>> &nodes) {
    for (auto it = nodes.begin(); it != nodes.end(); ++it) {
      const Ptr<MoralNode> &node = *it;
      auto other_it = it;
      ++other_it;
      for (; other_it != nodes.end(); ++other_it) {
        node->add_neighbor(*other_it);
      }
    }
  }

  //===========================================================================
  // Take an undirected graph of MoralNodes, potentially re-order them, and add
  // edges so that the graph is triangulated.
  void triangulate_moral_graph(std::vector<Ptr<MoralNode>> &nodes,
                               const std::function<double(Ptr<MoralNode>)> &criterion) {
    if (nodes.empty()) {
      return;
    }

    SortedVector<Ptr<MoralNode>> unnumbered_nodes;
    for (auto &el : nodes) {
      el->set_triangulation_number(-1);
      unnumbered_nodes.insert(el);
    }

    std::vector<std::set<Ptr<MoralNode>>> elimination_sets(nodes.size());

    // Note that nodes.size() is at least one, because we bail on entry if nodes
    // is empty.
    for (int i = nodes.size() - 1; i >= 0; --i) {
      Ptr<MoralNode> node = choose_node(unnumbered_nodes, criterion);
      node->set_triangulation_number(i);
      elimination_sets[i].insert(node);
      for (const auto &el : node->neighbors()) {
        Ptr<MoralNode> neighbor = node->promote(el);
        if (unnumbered_nodes.contains(neighbor)) {
          elimination_sets[i].insert(neighbor);
        }
      }
      make_dense(elimination_sets[i]);
    }

    make_elimination_tree(elimination_sets);
  }

  void make_elimination_tree(std::vector<std::set<Ptr<MoralNode>>>)

  void DirectedGraphicalModel::ensure_junction_tree() const {
    if (junction_tree_current_) {
      return;
    }

    std::vector<Ptr<MoralNode>> moral_nodes(create_moral_graph(
        std::vector<Ptr<DirectedNode>>(nodes_.begin(), nodes_.end())));

    triangulate_moral_graph(moral_nodes, triangulation_heuristic_);

    junction_tree_ = find_cliques(std::vector<Ptr<Node>>(
        moral_nodes.begin(), moral_nodes.end()));

    // Arrange cliques into a tree.

    // Given a collection of cliques, make them a tree.  Poof!

  }

}

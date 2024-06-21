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


  void DirectedGraphicalModel::ensure_junction_tree() const {
    if (junction_tree_current_) {
      return;
    }

    std::vector<Ptr<MoralNode>> moral_nodes(create_moral_graph(
        std::vector<Ptr<DirectedNode>>(nodes_.begin(), nodes_.end())));

    junction_tree_ = find_cliques(std::vector<Ptr<Node>>(
        moral_nodes.begin(), moral_nodes.end()));

    //     for (size_t i = 0; i <

  }

}

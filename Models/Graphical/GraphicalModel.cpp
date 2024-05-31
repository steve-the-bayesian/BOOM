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

#include "Models/Graphical/GraphicalModel.hpp"

namespace BOOM {

  void GraphicalModel::add_node(const Ptr<Node> &node) {
    nodes_.push_back(node);
    node_names_[node.name()] = node;
  }

  // TODO: Right now this is just the expression of logp for the fully observed
  // case.  We need to work out the case where the 
  double GraphicalModel::logp(const MixedMultivariateData &dp) const {
    double ans = 0;
    for (const auto &node : nodes_) {
      ans += node->logp(dp);
    }
    return ans;
  }

}  // namespace BOOM

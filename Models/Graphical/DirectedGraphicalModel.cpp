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
#include "Models/Graphical/Clique.hpp"
#include "Models/Graphical/DirectedGraphicalModel.hpp"
#include "cpputil/report_error.hpp"
#include <vector>
#include <map>
#include <functional>

namespace BOOM {
  using namespace Graphical;

  DirectedGraphicalModel::DirectedGraphicalModel()
      : junction_tree_current_(false)
  {}

  void DirectedGraphicalModel::add_node(const Ptr<Node> &node) {
    if (nodes_.contains(node)) {
      std::ostringstream err;
      err << "A node with id " << node->id() << " and name "
          << node->name() << " already exists in the model.";
      report_error(err.str());
    }

    nodes_.insert(node);
    junction_tree_current_ = false;
  }

  double DirectedGraphicalModel::logp(const MixedMultivariateData &dp) const {
    ensure_junction_tree();
    return junction_tree_.accumulate_evidence(dp);
  }


  //===========================================================================
  // If the junction tree has not yet been built, or if something has been
  // done to invalidate it, rebuild the tree.
  //
  // Effects:
  //   junction_tree_ is rebuilt if needed.
  //   junction_tree_current_ is set to true.
  void DirectedGraphicalModel::ensure_junction_tree() const {
    if (!junction_tree_current_) {
      junction_tree_.build(std::vector<Ptr<Node>>(
          nodes_.begin(), nodes_.end()));
      junction_tree_current_ = true;
    }
  }

}  // namespace BOOM

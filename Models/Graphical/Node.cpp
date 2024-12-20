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
#include <algorithm>
#include <sstream>
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace Graphical {

    namespace {
      bool is_in(const Ptr<Node> &element,
                 const std::vector<Ptr<Node>> &container) {
        return std::find(container.begin(),
                         container.end(),
                         element) != container.end();
      }

      bool is_in(const Node *element,
                 const std::vector<Ptr<Node>> &container) {
        return std::find_if(
            container.begin(),
            container.end(),
            [element](const Ptr<Node> &candidate) {
              return candidate.get() == element;
            }) != container.end();
      }
    }

    std::set<Ptr<Node>> Node::neighbors() const {
      std::set<Ptr<Node>> ans(other_neighbors_);
      for (const Ptr<Node> &parent : parents_) {
        ans.insert(parent);
      }
      for (const Ptr<Node> &child : children_) {
        ans.insert(child);
      }
      return ans;
    }

    void Node::add_parent(const Ptr<Node> &parent, bool reciprocate) {
      if (!is_in(parent, parents_)) {
        parents_.push_back(parent);
        if (reciprocate) {
          parent->add_child(this, false);
        }
      }
    }

    bool Node::is_parent(const Ptr<Node> &node) const {
      return is_in(this, node->parents());
    }

    void Node::add_child(const Ptr<Node> &child, bool reciprocate) {
      if (!is_in(child, children_)) {
        children_.push_back(child);
        if (reciprocate) {
          child->add_parent(this, false);
        }
      }
    }

    bool Node::is_child(const Ptr<Node> &node) const {
      return is_in(this, node->children());
    }

    bool Node::is_neighbor(const Ptr<Node> &node) const {
      return is_in(node, parents_) || is_in(node, children_);
    }

    double Node::numeric_value(
        const MixedMultivariateData &data_point) const {
      std::ostringstream err;
      err << "Node " << name() << " does not model numeric data.";
      report_error(err.str());
      return negative_infinity();
    }

    int Node::categorical_value(
        const MixedMultivariateData &data_point) const {
      std::ostringstream err;
      err << "Node " << name() << " does not model categorical data.";
      report_error(err.str());
      return -1;
    }

  }  // namespace Graphical
}  // namespace BOOM

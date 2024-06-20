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

namespace BOOM {
  namespace Graphical {

    namespace {
      bool is_in(const Ptr<DirectedNode> &element,
                 const std::vector<Ptr<DirectedNode>> &container) {
        return std::find(container.begin(),
                         container.end(),
                         element) != container.end();
      }

      bool is_in(const Node *element,
                 const std::vector<Ptr<DirectedNode>> &container) {
        return std::find_if(
            container.begin(),
            container.end(),
            [element](const Ptr<DirectedNode> &candidate) {
              return candidate.get() == element;
            }) != container.end();
      }
    }

    void DirectedNode::add_parent(const Ptr<DirectedNode> &parent, bool reciprocate) {
      if (!is_in(parent, parents_)) {
        parents_.push_back(parent);
        add_neighbor(parent, reciprocate);
        if (reciprocate) {
          parent->add_child(this, false);
        }
      }
    }

    bool DirectedNode::is_parent(const Ptr<DirectedNode> &node) const {
      return is_in(this, node->parents());
    }

    void DirectedNode::add_child(const Ptr<DirectedNode> &child, bool reciprocate) {
      if (!is_in(child, children_)) {
        children_.push_back(child);
        add_neighbor(child, reciprocate);
        if (reciprocate) {
          child->add_parent(this, false);
        }
      }
    }

    bool DirectedNode::is_child(const Ptr<DirectedNode> &node) const {
      return is_in(this, node->children());
    }

  }  // namespace Graphical
}  // namespace BOOM

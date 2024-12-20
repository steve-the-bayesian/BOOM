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

#include "Models/Graphical/Clique.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  namespace Graphical {

    Clique::Clique(const NodeSet &node_set) {
        for (const auto &el : node_set.elements()) {
          bool ok = try_add(el);
          if (!ok) {
            std::ostringstream err;
            err << "Could not add " << el->name()
                << " to the clique " << name();
            report_error(err.str());
          }
        }
      }

    bool Clique::try_add(const Ptr<Node> &node) {
      for (const auto &el : elements()) {
        if (!node->is_neighbor(el)) {
          return false;
        }
      }
      NodeSet::add(node);
      return true;
    }

    bool Clique::shares_node_with(const Ptr<Clique> &other) const {
      return !elements().disjoint_from(other->elements());
    }

    bool Clique::contains(const Ptr<Node> &node) const {
      for (const auto &el : elements()) {
        if (el == node) {
          return true;
        }
      }
      return false;
    }

  }  // namespace Graphical
} // namespace BOOM

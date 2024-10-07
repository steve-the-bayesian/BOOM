#ifndef BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_
#define BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_

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

#include <sstream>

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/NodeSet.hpp"

namespace BOOM {
  namespace Graphical {

    // A clique is NodeSet of Node's that are all neighbors of one another.
    class Clique : public NodeSet {
     public:
      // An empty Clique.
      Clique() {}

      // Create a Clique from a NodeSet.  All nodes in node_set must be
      // neighbors of one another.  Otherwise an exception is generated.
      Clique(const NodeSet &node_set);

      // Attempt to add a node, which may or may not belong, to the Clique.
      // Return true iff the addition was successful.
      //
      // Args:
      //   node:  A node to add to the Clique.
      //
      // Effects:
      //   If 'node' is a neighbor of all current elements, then 'node' is added
      //   to the set of elements_.
      //
      // Returns:
      //   A flag indicating whether 'node' was added to this object's elements.
      bool try_add(const Ptr<Node> &node);

      // Two cliques are equal if their elements_ are equal.
      bool operator==(const Clique &rhs) const {
        return elements() == rhs.elements();
      }

      // Return true iff *this and other have at least one element in common.
      bool shares_node_with(const Ptr<Clique> &other) const;

      // Returns true iff *this contains the specified node.
      bool contains(const Ptr<Node> &node) const;
    };


  } // namespace Graphical
}  // namespace BOOM

#endif  //  BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_

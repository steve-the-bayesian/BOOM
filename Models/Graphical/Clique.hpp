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

#include "Models/Graphical/Node.hpp"
#include <set>

namespace BOOM {
  namespace Graphical {

    // A clique is a collection of nodes that are all neighbors.  I.e. each node
    // is a neighbor of every other node in the clique.
    class Clique {
     public:
      // Attempt to add a node to the Clique.  Return true iff the addition was
      // successful
      bool try_add(const Ptr<Node> &node);

      // Two cliques are equal if their elements_ are equal.
      bool operator==(const Clique &rhs) const {
        return elements_ == rhs.elements_;
      }

     private:
      std::set<Ptr<Node>> elements_;
    };

  } // namespace Graphical
}  // namespace BOOM

#endif  //  BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_

#ifndef BOOM_MODELS_GRAPHICAL_NODE_HPP
#define BOOM_MODELS_GRAPHICAL_NODE_HPP

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

#include <cpputil/Ptr.hpp>
#include <cpputil/RefCounted.hpp>
#include <vector>


namespace BOOM {
  namespace Graphical {

    enum class NodeType{
      DISCRETE = 0,
      CONTINUOUS = 1,
    };

    class Node : private RefCounted {
     public:
      virtual std::vector<Ptr<Node>> parents() = 0;
      virtual std::vector<Ptr<Node>> children() = 0;
      virtual std::vector<Ptr<Node>> neighbors() = 0;
      virtual NodeType node_type() const = 0;

     private:
      friend void intrusive_ptr_add_ref(Node *d) { d->up_count(); }
      friend void intrusive_ptr_release(Node *d) {
        d->down_count();
        if (d->ref_count() == 0) delete d;
      }
    };

  }
}  // namespace BOOM


#endif  // BOOM_MODELS_GRAPHICAL_NODE_HPP

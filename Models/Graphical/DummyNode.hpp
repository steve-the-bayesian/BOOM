#ifndef BOOM_GRAPHICAL_MODELS_DUMMY_NODE_HPP_
#define BOOM_GRAPHICAL_MODELS_DUMMY_NODE_HPP_

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

namespace BOOM {

  namespace Graphical {
    class DummyNode : public Node {
     public:

      DummyNode(int id, const std::string &name = "")
          : Node(id, name)
      {}

      NodeType node_type() const override {
        return NodeType::DUMMY;
      }

      Int dim() const override {
        return -1;
      }

      double logp(const MixedMultivariateData &dp) const override {
        return 0.0;
      }
    };

  }  // namespace Graphical

}  // namespace BOOM

#endif  // BOOM_GRAPHICAL_MODELS_DUMMY_NODE_HPP_

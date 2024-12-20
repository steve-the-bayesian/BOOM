#ifndef BOOM_MODELS_GRAPHICAL_CATEGORICAL_NODE_HPP_
#define BOOM_MODELS_GRAPHICAL_CATEGORICAL_NODE_HPP_

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

namespace BOOM {
  namespace Graphical {

    class CategoricalNode : public Node {
     public:
      CategoricalNode(const Ptr<CatKeyBase> &values);
      // Other constructors as needed

      NodeType node_type() const override {
        return NodeType::DISCRETE;
      }

     private:
      Ptr<CatKeyBase> values_;

      // A flag indicating whether the conditional probability tables are
      // currently valid.  A CPT can become invalid if a new parent or neighbor
      // is added to the node.
      mutable bool current_;


    };

  }  // namespace Graphical
}  // namespace BOOM

#endif // BOOM_MODELS_GRAPHICAL_CATEGORICAL_NODE_HPP_

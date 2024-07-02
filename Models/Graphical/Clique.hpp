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
#include "cpputil/SortedVector.hpp"

namespace BOOM {
  namespace Graphical {

    // A clique is a collection of nodes that are all neighbors.  I.e. each node
    // is a neighbor of every other node in the clique.
    class Clique : public Node {
     public:

      Clique(int id, const std::string &name = "")
          : Node(id, name)
      {}

      const std::string &name() const override;

      Clique * promote(const Ptr<Node> &rhs) {
        return rhs->reflect_clique();
      }
      const Clique * promote_const(const Ptr<Node> &rhs) const {
        return rhs->reflect_const_clique();
      }

      Clique * reflect_clique() override {
        return this;
      }
      const Clique * reflect_const_clique() const override {
        return this;
      }

      // Attempt to add a node, which may or may not belong, to the Clique.
      // Return true iff the addition was successful.
      bool try_add(const Ptr<Node> &node);

      // Two cliques are equal if their elements_ are equal.
      bool operator==(const Clique &rhs) const {
        return elements_ == rhs.elements_;
      }

      std::string print() const;

      size_t size() const {
        return elements_.size();
      }

      const SortedVector<Ptr<Node>> &elements() const {
        return elements_;
      }

      bool shares_node_with(const Ptr<Clique> &other) const {
        return elements_.disjoint_from(other->elements_);
      }

     private:
      SortedVector<Ptr<Node>> elements_;
    };

    std::vector<Ptr<Clique>> find_cliques(const std::vector<Ptr<Node>> &nodes);

    class CliqueTree {
     public:
      int y;

     private:
      std::vector<Ptr<Clique>> elements_;
    };

  } // namespace Graphical
}  // namespace BOOM

#endif  //  BOOM_GRAPHICAL_MODELS_CLIQUE_HPP_

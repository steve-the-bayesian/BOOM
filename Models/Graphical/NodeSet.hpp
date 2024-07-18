#ifndef BOOM_GRAPHICAL_NODESET_HPP_
#define BOOM_GRAPHICAL_NODESET_HPP_
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

#include <sstream>

namespace BOOM {

  namespace Graphical {
    // A NodeSet is a set of nodes, but it is also a node in the "hypergraph" of
    // sets of nodes.  If a NodeSet is complete (all its elements are neighbors
    // of one another) then it can be promoted to a clique.
    template <class NODETYPE>
    class NodeSet : public Node {
     public:
      // The NodeSet starts with an empty value for both its id and name.  The
      // name gets modified when nodes are added to the set.  The id is
      // available to use for different numberings needed by triangulation
      // algorithms.
      using iterator = typename SortedVector<Ptr<NODETYPE>>::iterator;
      using const_iterator = typename SortedVector<Ptr<NODETYPE>>::const_iterator;

      NodeSet()
          : Node(-1, "")
      {}

      // Set the numeric id of the node to a given value.
      void set_id(int id) {
        Node::set_id(id);
      }

      NodeSet(const SortedVector<NODETYPE> &elements)
          : Node(-1, ""),
            elements_(elements)
      {}

      //-----------------------------------------------------------------------
      // Set interface

      // Add a node to the set.
      void add(const Ptr<NODETYPE> &node) {
        elements_.insert(node);
        recompute_name();
      }

      // Syntactic sugar.
      void insert(const Ptr<NODETYPE> &node) { add(node); }

      const SortedVector<Ptr<NODETYPE>> &elements() const {
        return elements_;
      }

      //-----------------------------------------------------------------------
      // Container interface
      iterator begin() { return elements_.begin();}
      iterator end() { return elements_.end();}
      const_iterator begin() const { return elements_.begin();}
      const_iterator end() const { return elements_.end();}
      size_t size() const {return elements_.size();}

      //-----------------------------------------------------------------------
      // NodeSet as a node in a hypergraph.
      void add_neighbor(const Ptr<NodeSet<NODETYPE>> &nodeset,
                        bool reciprocate = true) {
        neighbors_.insert(nodeset);
        if (reciprocate) {
          // Don't reciprocate the reciprocation, or there will be an infinite
          // loop.
          nodeset->add_neighbor(this, false);
        }
      }

      const std::set<Ptr<NodeSet<NODETYPE>>> &neighbors() const {
        return neighbors_;
      }

      bool is_subset(const NodeSet<NODETYPE> &other) const {
        return elements_.is_subset(other.elements_);
      }

      // Absorb the elements of other into *this;
      void absorb(const NodeSet<NODETYPE> &other) {
        elements_.absorb(other.elements_);
      }

      NodeSet<NODETYPE> intersection(const NodeSet<NODETYPE> &other) const {
        return NodeSet(elements_.intersection(other.elements_));
      }

     private:
      SortedVector<Ptr<NODETYPE>> elements_;
      std::set<Ptr<Graphical::NodeSet<NODETYPE>>> neighbors_;

      // The name of the NodeSet is the name of the
      void recompute_name() {
        std::ostringstream name;
        bool start = true;
        for (auto &el : elements_) {
          if (start) {
            start = false;
          } else {
            name << ":";
          }
          name << el->name();
        }
        set_name(name.str());
      }
    };


  }  // namespace Graphical

}  // namespace BOOM


#endif //  BOOM_GRAPHICAL_NODESET_HPP_

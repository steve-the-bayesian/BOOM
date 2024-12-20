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
#include "cpputil/RefCounted.hpp"

#include <sstream>
#include <initializer_list>

namespace BOOM {

  namespace Graphical {
    // A NodeSet is a set of nodes, but it is also a node in the "hypergraph" of
    // sets of nodes.  If a NodeSet is complete (all its elements are neighbors
    // of one another) then it can be promoted to a clique.
    class NodeSet : private RefCounted {
     public:
      // The NodeSet starts with an empty value for both its id and name.  The
      // name gets modified when nodes are added to the set.  The id is
      // available to use for different numberings needed by triangulation
      // algorithms.

      friend void intrusive_ptr_add_ref(NodeSet *n) {n->up_count();}
      friend void intrusive_ptr_release(NodeSet *n) {
        n->down_count();
        if (n->ref_count() == 0) {
          delete n;
        }
      }

      using iterator = typename SortedVector<Ptr<Node>>::iterator;
      using const_iterator = typename SortedVector<Ptr<Node>>::const_iterator;

      NodeSet()
          : id_(-1)
      {}

      NodeSet(const std::initializer_list<Ptr<Node>> &list)
          : id_(-1)
      {
        for(auto el : list) {
          add(el);
        }
      }

      template <class ITERATOR>
      NodeSet(ITERATOR begin, ITERATOR end)
          : id_(-1)
      {
        for (ITERATOR it = begin; it != end; ++it) {
          insert(*it);
        }
      }

      explicit NodeSet(const SortedVector<Ptr<Node>> &elements)
          : id_(-1),
            elements_(elements)
      {}

      // The 'id' of a NodeSet is typically a numbering produced by a
      // triangulation algorithm when forming a tree of cliques.
      void set_id(int id) {
        id_ = id;
      }

      int id() const {
        return id_;
      }

      std::ostream &print(std::ostream &out) const {
        out << "[";
        for (const auto &node : elements_) {
          out << *node << ' ';
        }
        out << "]";
        return out;
      }

      int index (const Ptr<Node> &node) const {
        return elements_.position(node);
      }

      //-----------------------------------------------------------------------
      // Set interface
      //-----------------------------------------------------------------------

      // Add a node to the set.
      void add(const Ptr<Node> &node) {
        elements_.insert(node);
      }

      // Syntactic sugar for 'add'.
      void insert(const Ptr<Node> &node) { add(node); }

      const SortedVector<Ptr<Node>> &elements() const {
        return elements_;
      }

      bool is_subset(const NodeSet &other) const {
        return elements_.is_subset(other.elements_);
      }

      bool contains(const Ptr<Node> &node) const {
        return elements_.contains(node);
      }

      // Absorb the elements of other into *this;
      void absorb(const NodeSet &other) {
        elements_.absorb(other.elements_);
      }

      NodeSet intersection(const NodeSet &other) const {
        SortedVector<Ptr<Node>> intersection_elements
            = elements_.intersection(other.elements_);
        return NodeSet(intersection_elements.begin(), intersection_elements.end());
      }

      //-----------------------------------------------------------------------
      // Container interface
      //-----------------------------------------------------------------------
      iterator begin() { return elements_.begin();}
      iterator end() { return elements_.end();}
      const_iterator begin() const { return elements_.begin();}
      const_iterator end() const { return elements_.end();}
      size_t size() const {return elements_.size();}
      void clear() {elements_.clear();}
      bool empty() const {return elements_.empty();}

      NodeSet::const_iterator remove(const Ptr<Node> &element) {
        return elements_.remove(element);
      }

      std::string name() const {
        return compute_name();
      }

      Ptr<Node> & operator[](size_t i) {
        return elements_[i];
      }

      const Ptr<Node> & operator[](size_t i) const {
        return elements_[i];
      }

      bool operator==(const NodeSet &rhs) const {
        if (elements_.size() != rhs.elements_.size()) {
          return false;
        }

        for (int i = 0; i < elements_.size(); ++i) {
          if (elements_[i] != rhs.elements_[i]) {
            return false;
          }
        }
        return true;
      }

     private:
      int id_;
      SortedVector<Ptr<Node>> elements_;

      // The name of the NodeSet is the name of the
      std::string compute_name() const {
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
        return name.str();
      }

    };

    inline std::ostream &operator<<(std::ostream &out,
                                    const NodeSet &node_set) {
      return node_set.print(out);
    }

  }  // namespace Graphical

}  // namespace BOOM


#endif //  BOOM_GRAPHICAL_NODESET_HPP_

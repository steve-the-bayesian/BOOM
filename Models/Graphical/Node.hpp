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

#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"

#include <vector>
#include <set>
#include <string>

namespace BOOM {
  namespace Graphical {

    // The types of variables that can be modeled by a probabilistic graphical
    // model.
    enum class NodeType{
      DUMMY = -1,
      CATEGORICAL = 0,
      CONTINUOUS = 1,
      ID = 2,
      DATETIME = 3,
    };

    //===========================================================================
    // A Node is an undirected node in a graph.  In the context of graphical
    // models, it may represent a variable (e.g. in a Markov random field) or a
    // Clique in a moral graph of of BayesNet.
    //
    // This code assumes that all Node objects are held in a Ptr.
    class Node : private RefCounted {
     public:

      // Args:
      //   id:  An index uniquely identifying the node in the graph.
      //   name:  A string, intended for human consumption.
      Node(int node_id, const std::string &name = "")
          : id_(node_id),
            name_(name)
      {}

      // The node's unique identifier in the graph.
      int id() const {return id_;}

      // An optional human-interpretable string indicating the node's relevance.
      virtual const std::string & name() const {return name_;}

      //---------------------------------------------------------------------------
      // Parents, children, neighbors.
      //---------------------------------------------------------------------------
      void add_neighbor(const Ptr<Node> &node, bool reciprocate = true) {
        neighbors_.insert(node);
        if (reciprocate) {
          node->add_neighbor(this, false);
        }
      }

      std::set<Ptr<Node>> neighbors() const {
        return neighbors_;
      }

      bool is_neighbor(const Ptr<Node> &node) const {
        return neighbors_.count(node);
      }

     private:
      friend void intrusive_ptr_add_ref(Node *d) { d->up_count(); }
      friend void intrusive_ptr_release(Node *d) {
        d->down_count();
        if (d->ref_count() == 0) delete d;
      }

      int id_;
      std::string name_;

      std::set<Ptr<Node>> neighbors_;
    };

    //===========================================================================
    // A DirectedNode in a GraphicalModel represents a variable (i.e. a column
    // in a data frame).
    class DirectedNode : public Node {
     public:
      DirectedNode(int id, const std::string &name = "")
          : Node(id, name)
      {}

      virtual NodeType node_type() const = 0;

      // Args:
      //   parent:  A parent of this this node.
      //   reciprocate: If true then this node will be added as a child of
      //     parent (modifying parent).
      //
      // Side Effects:
      //   The parent node is also added as a neighbor.
      void add_parent(const Ptr<DirectedNode> &parent, bool reciprocate = true);

      const std::vector<Ptr<DirectedNode>> & parents() const {
        return parents_;
      }

      // Return true iff this is a parent of 'node'.
      bool is_parent(const Ptr<DirectedNode> &node) const;

      // Args:
      //   child:  A child of this this node.
      //   reciprocate: If true then this node will be added as a parent of
      //     child (modifying child).
      //
      // Side Effects:
      //   The child node is also added as a neighbor.
      void add_child(const Ptr<DirectedNode> &child, bool reciprocate = true);

      const std::vector<Ptr<DirectedNode>> & children() const {
        return children_;
      }

      // Return true iff 'this' is a child of 'node'.
      bool is_child(const Ptr<DirectedNode> &node) const;


     private:
      std::vector<Ptr<DirectedNode>> parents_;
      std::vector<Ptr<DirectedNode>> children_;

    };

  }
}  // namespace BOOM


#endif  // BOOM_MODELS_GRAPHICAL_NODE_HPP

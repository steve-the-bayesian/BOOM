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

#include "stats/DataTable.hpp"  // contains DataTable and MixedMultivariateData.

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

    class DirectedNode;
    class MoralNode;
    class Clique;

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

      // A double dispatch method for promoting a base class node to a node of a
      // specific type.  Faster than casting.  Each concrete node class will
      // have a reflect method that returns 'this' if called from the same
      // class, and a pointer to Node if not.
      virtual Node *promote(const Ptr<Node> &rhs) = 0;
      virtual const Node *promote_const(const Ptr<Node> &rhs) const = 0;

      virtual DirectedNode *reflect_directed_node() {
        return nullptr;
      }
      virtual const DirectedNode *reflect_const_directed_node() const {
        return nullptr;
      }

      virtual MoralNode *reflect_moral_node() {
        return nullptr;
      }
      virtual const MoralNode *reflect_const_moral_node() const {
        return nullptr;
      }

      virtual Clique *reflect_clique() {
        return nullptr;
      }
      virtual const Clique *reflect_const_clique() const {
        return nullptr;
      }

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

     protected:
      void set_name(const std::string &name) const {
        name_ = name;
      }

      void set_id(int id) {
        id_ = id;
      }

     private:
      friend void intrusive_ptr_add_ref(Node *d) { d->up_count(); }
      friend void intrusive_ptr_release(Node *d) {
        d->down_count();
        if (d->ref_count() == 0) delete d;
      }

      int id_;
      mutable std::string name_;

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

      // This node's contribution to the log density of dp.  The conditional
      // distribution of this node's chunk of dp, given its parents.
      //
      // The Node knows how to pick out the relevant bits of dp, as well as pass
      // the relevant bits to its parents for conditioning.
      virtual double logp(const MixedMultivariateData &dp) const = 0;

      DirectedNode * promote(const Ptr<Node> &rhs) override {
        return rhs->reflect_directed_node();
      }
      const DirectedNode * promote_const(const Ptr<Node> &rhs) const override {
        return rhs->reflect_const_directed_node();
      }

      DirectedNode * reflect_directed_node() override {
        return this;
      }
      const DirectedNode * reflect_const_directed_node() const override {
        return this;
      }

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

    //===========================================================================
    // A MoralNode is a node in an undirected graph obtained by taking
    // DirectedNode's, marrying the parents, and dropping the arrows.
    class MoralNode : public Node {
     public:
      MoralNode(const Ptr<DirectedNode> &base_node)
          : Node(base_node->id(), base_node->name()),
            base_node_(base_node),
            triangulation_number_(-1)
      {}

      const Ptr<DirectedNode> &base_node() const {
        return base_node_;
      }

      MoralNode *promote(const Ptr<Node> &rhs) override {
        return rhs->reflect_moral_node();
      }
      const MoralNode *promote_const(const Ptr<Node> &rhs) const override {
        return rhs->reflect_const_moral_node();
      }

      MoralNode *reflect_moral_node() {
        return this;
      }
      const MoralNode *reflect_const_moral_node() const {
        return this;
      }

      // The 'triangulation_number' is a node ordering device that is part of an
      // algorithm for triangulating the moral graph.  See Cowell et al. p 58.
      int triangulation_number() const {
        return triangulation_number_;
      }

      void set_triangulation_number(int number) {
        triangulation_number_ = number;
      }

     private:
      Ptr<DirectedNode> base_node_;
      int triangulation_number_;
    };

  }
}  // namespace BOOM


#endif  // BOOM_MODELS_GRAPHICAL_NODE_HPP

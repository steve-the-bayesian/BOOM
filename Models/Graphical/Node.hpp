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

    };

    // Order nodes by their ID.
    struct IdLess {
      bool operator()(const Ptr<::BOOM::Graphical::Node> &n1,
                      const Ptr<::BOOM::Graphical::Node> &n2) const {
        return n1->id() < n2->id();
      }
    };

    //===========================================================================
    // A DirectedNode in a GraphicalModel represents a variable (i.e. a column
    // in a data frame).
    class DirectedNode : public Node {
     public:
      // Args:
      //   id: The position in a data frame or MixedMultivariateData where this
      //     node's variable can be found.
      //   name: The column name in a data frame, or name in a
      //     MixedMultivariateData, describing the variable to be modeled.
      DirectedNode(int id, const std::string &name = "")
          : Node(id, name),
            variable_index_(id)
      {}

      // Find the position of a variable with name matching the name of this node.
      // If no such variable is found, then the variable's index is set to -1.
      //
      // Args:
      //   data_point: The data to search for the relevant variable name.
      //   throw_on_error: If true, then an exception is thrown if the variable
      //     name is not found.
      void find_variable(const MixedMultivariateData &data_point,
                         bool throw_on_error = true);

      virtual NodeType node_type() const = 0;

      // This node's contribution to the log density of dp.  The conditional
      // distribution of this node's chunk of dp, given its parents.
      //
      // The Node knows how to pick out the relevant bits of dp, as well as pass
      // the relevant bits to its parents for conditioning.
      virtual double logp(const MixedMultivariateData &dp) const = 0;

      bool is_variable_observed(const MixedMultivariateData &data_point) const {
        return data_point.variable(variable_index_).missing() == Data::observed;
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

      bool is_neighbor(const Ptr<DirectedNode> &node) const;

     private:
      std::vector<Ptr<DirectedNode>> parents_;
      std::vector<Ptr<DirectedNode>> children_;

      // The position in a data frame or MixedMultivariateData where this node's
      // variable is found.
      int variable_index_;
    };

    //===========================================================================
    // A MoralNode is a node in an undirected graph obtained by taking
    // DirectedNode's, marrying the parents, and dropping the arrows.
    class MoralNode : public Node {
     public:
      MoralNode(const Ptr<DirectedNode> &base_node)
          : Node(base_node->id(), base_node->name()),
            base_node_(base_node)
      {}

      const Ptr<DirectedNode> &base_node() const {
        return base_node_;
      }

      void set_id(int id) { Node::set_id(id); }

      void add_neighbor(const Ptr<MoralNode> &node, bool reciprocate = true) {
        neighbors_.insert(node);
        if (reciprocate) {
          node->add_neighbor(this, false);
        }
      }

      std::set<Ptr<MoralNode>> neighbors() const {
        return neighbors_;
      }

      bool is_neighbor(const Ptr<MoralNode> &node) const {
        return neighbors_.count(node);
      }

     private:
      Ptr<DirectedNode> base_node_;

      std::set<Ptr<MoralNode>> neighbors_;
    };

  } // namespace Graphical
}  // namespace BOOM


#endif  // BOOM_MODELS_GRAPHICAL_NODE_HPP

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
      NUMERIC = 1,
      ID = 2,
      DATETIME = 3,
    };

    //===========================================================================
    // A Node is a node in a graph.  In the context of graphical
    // models, it may represent a variable (e.g. in a Markov random field) or a
    // Clique in a moral graph of of BayesNet.
    //
    // This code assumes that all Node objects are held in a Ptr.
    class Node : private RefCounted {
     public:
      // Args:
      //   id: An index uniquely identifying the node in the graph.  Categorical
      //     variables should have smaller numeric id's than numeric variables.
      //   name: A string, intended for human consumption.  Often the name of
      //     the variable modeled by the node.
      //   variable_index: The position (column number) of the variable being
      //     modeled in the data file.
      explicit Node(int node_id, const std::string &name = "", int variable_index = -1)
          : id_(node_id),
            name_(name),
            variable_index_(variable_index)
      {}

      // An optional human-interpretable string indicating the node's identity.
      virtual const std::string & name() const {return name_;}

      // Change the node's human readable name.
      void set_name(const std::string &name) const {
        name_ = name;
      }

      // The node's numeric unique identifier in the graph.
      int id() const {return id_;}

      // Change the node's numeric id.
      void set_id(int id) {
        id_ = id;
      }

      std::ostream &print(std::ostream &out) const {
        out << id() << ' ' << name() << " |";
        for (const auto &child : children_) {
          out << child->id() << ' ' << child->name();
        }
        return out;
      }

      //---------------------------------------------------------------------------
      // Graph relationships.  Parents, children, other neighbors.
      //---------------------------------------------------------------------------

      // Args:
      //   parent:  A parent of this this node.
      //   reciprocate: If true then this node will be added as a child of
      //     parent (modifying parent).
      //
      // Side Effects:
      //   The parent node is also added as a neighbor.
      void add_parent(const Ptr<Node> &parent, bool reciprocate = true);

      const std::vector<Ptr<Node>> & parents() const {
        return parents_;
      }

      // Returns true iff node is either a parent or child of *this, or in
      // other_neighhbors_.
      bool is_neighbor(const Ptr<Node> &node) const;

      // Return true iff this is a parent of 'node'.
      bool is_parent(const Ptr<Node> &node) const;

      // Args:
      //   child:  A child of this this node.
      //   reciprocate: If true then this node will be added as a parent of
      //     child (modifying child).
      //
      // Side Effects:
      //   The child node is also added as a neighbor.
      void add_child(const Ptr<Node> &child, bool reciprocate = true);

      const std::vector<Ptr<Node>> & children() const {
        return children_;
      }

      // Return true iff 'this' is a child of 'node'.
      bool is_child(const Ptr<Node> &node) const;

      // Add an undirected neighbor relationship.
      // Args:
      //   node:  The node to make a neighbor of 'this' node.
      //   reciprocate:  If true then also add 'this' as a neighbor of 'node'.
      void add_neighbor(const Ptr<Node> &node, bool reciprocate = true) {
        other_neighbors_.insert(node);
        if (reciprocate) {
          node->add_neighbor(this, false);
        }
      }

      // Return the set of all neighbors of this node, including all parents,
      // children, and other neighbors.
      std::set<Ptr<Node>> neighbors() const;

      // Remove all undirected "other" neighbor relationships.
      void clear_neighbors() {
        other_neighbors_.clear();
      }

      //---------------------------------------------------------------------------
      // Relationships between the node and the data it models.
      //---------------------------------------------------------------------------

      // Find the position of a variable with name matching the name of this node.
      // If no such variable is found, then the variable's index is set to -1.
      //
      // Args:
      //   data_point: The data to search for the relevant variable name.
      //   throw_on_error: If true, then an exception is thrown if the variable
      //     name is not found.
      // void find_variable(const MixedMultivariateData &data_point,
      //                    bool throw_on_error = true);

      // The column number in the data table of the variable being modeled by this node.
      int variable_index() const {return variable_index_;}
      void set_variable_index(int index) { variable_index_ = index; }

      virtual NodeType node_type() const = 0;

      // The dimension of the variable being modeled.  For categorical data this
      // is the number of levels.  For numeric data the dimension is 1.
      virtual Int dim() const = 0;

      // This node's contribution to the log density of dp.  The conditional
      // distribution of this node's chunk of dp, given its parents.
      //
      // The Node knows how to pick out the relevant bits of dp, as well as pass
      // the relevant bits to its parents for conditioning.
      virtual double logp(const MixedMultivariateData &dp) const = 0;

      bool is_missing(const MixedMultivariateData &data_point) const {
        return data_point.variable(variable_index_).missing() != Data::observed;
      }

      bool is_observed(const MixedMultivariateData &data_point) const {
        return !is_missing(data_point);
      }

      // Return the observed data value for this node's variable.  If the node
      // does not model numeric data then throw an exception.  If the value is
      // missing return NaN.
      virtual double numeric_value(
          const MixedMultivariateData &data_point) const;

      // Return the observed data value for this node's variable.  If the node
      // does not model categorical data then throw an exception.  If the value
      // is missing return -1.
      virtual int categorical_value(
          const MixedMultivariateData &data_point) const;

     private:
      friend void intrusive_ptr_add_ref(Node *d) { d->up_count(); }
      friend void intrusive_ptr_release(Node *d) {
        d->down_count();
        if (d->ref_count() == 0) delete d;
      }

      int id_;
      mutable std::string name_;

      std::vector<Ptr<Node>> parents_;
      std::vector<Ptr<Node>> children_;

      // For directed graphs, all neighboring nodes will either be parents or
      // children.  For chain graphs there can be undirected links as well.  The
      // main source of neighboring links is graph moralization or triangulation
      // when forming the junction tree.
      std::set<Ptr<Node>> other_neighbors_;

      // The position in a data frame or MixedMultivariateData where this node's
      // variable is found.
      int variable_index_;
    };

    inline std::ostream & operator<<(std::ostream &out, const Ptr<Node> &node) {
      return node->print(out);
    }

    inline std::ostream & operator<<(std::ostream &out, const Node *node) {
      return node->print(out);
    }

    inline std::ostream & operator<<(std::ostream &out, const Node &node) {
      return node.print(out);
    }

  } // namespace Graphical
}  // namespace BOOM


#endif  // BOOM_MODELS_GRAPHICAL_NODE_HPP

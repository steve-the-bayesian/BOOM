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

#include "Models/Graphical/NodeSetMarginalDistribution.hpp"

namespace BOOM {
  namespace Graphical {

    void intrusive_ptr_add_ref(NodeSetMarginalDistribution *d) {
      d->up_count();
    }

    void intrusive_ptr_release(NodeSetMarginalDistribution *d) {
      d->down_count();
      if (d->ref_count() == 0) delete d;
    }

    namespace {
      using NSMD = NodeSetMarginalDistribution;


      inline bool is_in(const Ptr<Node> &node,
                        const std::vector<Ptr<Node>> &nodes) {
        return std::find(nodes.begin(), nodes.end(), node) != nodes.end();
      }
    }  // namespace

    //===========================================================================
    NSMD::NodeSetMarginalDistribution(const NodeSet *nodes)
        : host_(nodes)
    {}

    //===========================================================================
    // Store values for the known nodes.
    void NSMD::resize(const MixedMultivariateData &data_point) {
      known_discrete_variables_.clear();
      known_gaussian_variables_.clear();
      unknown_discrete_nodes_.clear();
      unknown_gaussian_nodes_.clear();
      std::vector<int> unknown_dims;

      for (const auto &node : host_->elements()) {
        if (node->is_observed(data_point)) {
          // If the node's variable is observed, then store it in the
          // appropriate map.
          //
          // TODO: this code might be clearer if Node types were
          // subclassed in the type system.
          switch (node->node_type()) {
            case NodeType::CATEGORICAL:
              known_discrete_variables_[node] = node->categorical_value(
                  data_point);
              break;

            case NodeType::CONTINUOUS:
              known_gaussian_variables_[node] = node->numeric_value(
                  data_point);
              break;

            default:
              report_error("Unexpected case.");
          }
        } else {
          if (node->node_type() == NodeType::CATEGORICAL) {
            // Here the node's value is unobserved.  Store the node in
            // unknown_discrete_nodes_, and add its dimension to unknown_dims.
            unknown_discrete_nodes_.insert(node);
            unknown_dims.push_back(node->dim());
          } else {
            report_error("Only categorical variables handled for now.");
          }
        }
      }

      unknown_discrete_distribution_ = Array(unknown_dims, 0.0);
    }

    //===========================================================================
    bool NSMD::is_known(const Ptr<Node> &directed) const {
      if (directed->node_type() == NodeType::CATEGORICAL) {
        auto it = known_discrete_variables_.find(directed);
        return it != known_discrete_variables_.end();
      } else if (directed->node_type() == NodeType::CONTINUOUS) {
        auto it = known_gaussian_variables_.find(directed);
        return it != known_gaussian_variables_.end();
      } else {
        std::ostringstream err;
        err << "Node " << directed->name()
            << " is neither categorical nor continuous.";
        report_error(err.str());
      }
      return false;
    }

    // =========================================================================
    // Args:
    //   data_point: The data point containing evidence to be distributed.
    //   parent_distribution: The distribution over the unknown values in the
    //     parent node set in the junction tree.  If this node is a root of the
    //     junction tree then parent_distribution is nullptr.
    //
    // Returns:
    //   The "observed data log likelihood" contribution of the new information
    //   observed in this node set.
    //
    // Effects:
    //   The marginal distribution of the unknown variables is populated.  Upon
    //   exit it contains the conditional distribution of all the unknown
    //   variables in the node set given all the known variables in this node set
    //   and node sets in ancestor nodes on the junction tree.
    double NSMD::forward_increment(
        const MixedMultivariateData &data_point,
        const NodeSetMarginalDistribution *parent_distribution) {
      resize(data_point);

      // 1) Find all the nodes in the d-separator.  These have already been
      //    processed.  Separate them out into knowns vs unknowns, and find the
      //    marginal distribution of the unknowns.
      NodeSet d_separator;
      NodeSetMarginalDistribution prior_margin(&d_separator);
      if (parent_distribution) {
        d_separator = parent_distribution->host()->intersection(*host_);
        prior_margin = parent_distribution->compute_margin(d_separator);
      } else {
        // Is there anything to do here?
      }



      // 2) Find all the children of the nodes in the d-separator, as well as
      //    all the nodes that have no parents.  Process the unknown nodes.
      //    Update the distribution conditional on the known nodes.

      // 3) Repeat step 2 for all the nodes that have parents in the processed
      //    set (adding nodes to the processed set as they are processed).  Keep
      //    repeating until all nodes have been processed.
      return 0.0;
    }

    //===========================================================================

    // Compute the marginal distribution for the nodes in the subset.
    //
    // The known variables in the subset are copied.  The unknown variables NOT
    // in the subset are summed over.
    NSMD NSMD::compute_margin(const NodeSet &subset) const {

      NodeSetMarginalDistribution ans(&subset);
      NodeSet sum_over_nodes(unknown_discrete_nodes_);

      for (const auto &node : subset) {
        if (!host_->contains(node)) {
          std::ostringstream err;
          err << "NodeSetMarginalDistribution for " << *host_
              << " was asked to compute a margin containing "
              << *node
              << ", which is not part of the node set.";
          report_error(err.str());
        } else {
          switch(node->node_type()) {
            case NodeType::CATEGORICAL:
              if (is_known(node)) {
                ans.known_discrete_variables_[node]
                    = this->known_discrete_variables_.find(node)->second;
              } else {
                sum_over_nodes.remove(node);
                ans.unknown_discrete_nodes_.add(node);
              }
              break;

            case NodeType::CONTINUOUS:
              if (is_known(node)) {
                ans.known_gaussian_variables_[node]
                    = known_gaussian_variables_.find(node)->second;
              } else {
                std::ostringstream err;
                err << node->name() << " is an unknown Gaussian node.  Support"
                    << " for unknown Gaussian nodes has not yet been "
                    << "implemented.";
                report_error(err.str());
              }
              break;

            case NodeType::DUMMY:
              {
                std::ostringstream err;
                err << "Dummy node encountered:"  << node->name();
                report_warning(err.str());
                break;
              }

            case NodeType::ID:
              {
                std::ostringstream err;
                err << "ID node encountered.  ID nodes are not yet supported.";
                report_error(err.str());
                break;
              }
            default:
              break;
          }
        }
      }

      std::vector<int> sum_over_dims;
      for (const auto &node : sum_over_nodes) {
        // Need the dimension in the array corresponding to this node.
        int index = unknown_discrete_nodes_.index(node);
        if (index < 0) {
          std::ostringstream err;
          err << "Node " << node->name() << " is to be marginalized over, "
              "but is not present in unknown_discrete_nodes_.";
          report_error(err.str());
        }
        sum_over_dims.push_back(unknown_discrete_nodes_.index(node));
      }

      ans.unknown_discrete_distribution_
          = unknown_discrete_distribution_.sum(sum_over_dims);

      return ans;
    }


  }  // namespace Graphical
}  // namespace BOOM

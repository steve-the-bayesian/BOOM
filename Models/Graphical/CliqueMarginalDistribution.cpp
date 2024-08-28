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

#include "Models/Graphical/CliqueMarginalDistribution.hpp"

namespace BOOM {
  namespace Graphical {

    namespace {
      using CMD = CliqueMarginalDistribution;
    }  // namespace

    CMD::CliqueMarginalDistribution(Clique *clique)
        : host_(clique)
    {}

    void CMD::resize(const MixedMultivariateData &data_point) {
      known_discrete_variables_.clear();
      known_gaussian_variables_.clear();
      unknown_nodes_.clear();
      std::vector<int> unknown_dims;

      for (const auto &node : host_->elements()) {
        Ptr<DirectedNode> base = node->base_node();
        if (base->is_observed(data_point)) {
          // If the node's variable is observed, then store it in the
          // appropriate map.
          //
          // TODO: this code might be clearer if DirectedNode types were
          // subclassed in the type system.
          switch (base->node_type()) {
            case NodeType::CATEGORICAL:
              known_discrete_variables_[base] = base->categorical_value(
                  data_point);
              break;

            case NodeType::CONTINUOUS:
              known_gaussian_variables_[base] = base->numeric_value(
                  data_point);
              break;

            default:
              report_error("Unexpected case.");
          }
        } else {
          // Here the node's value is unobserved.  Store the node in
          // unknown_nodes_, and add its dimension to unknown_dims.
          if (base->node_type() == NodeType::CATEGORICAL) {
            unknown_nodes_.push_back(base);
            unknown_dims.push_back(base->dim());
          } else {
            report_error("Only categorical variables handled for now.");
          }
        }
      }

      unknown_discrete_distribution_ = Array(unknown_dims, 0.0);
    }


    bool CMD::is_known(const Ptr<DirectedNode> &directed) const {
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

    // Args:
    //   data_point: The data point containing evidence to be distributed.
    //   parent_distribution: The distribution over the unknown values in the
    //     parent clique in the junction tree.  If this node is a root of the
    //     junction tree then parent_distribution is nullptr.
    //
    // Returns:
    //   The "observed data log likelihood" contribution of the new information
    //   observed in this clique.
    double CMD::forward_increment(
        const MixedMultivariateData &data_point,
        const CliqueMarginalDistribution *parent_distribution) {
      resize(data_point);

      // Get all the parents of all the nodes in the clique.

      std::vector<Ptr<DirectedNode>> parent_nodes;
      std::set<Ptr<DirectedNode>> unique_parent_nodes;
      std::vector<int> unknown_parent_dims;
      std::set<Ptr<DirectedNode>> known_parents;
      std::vector<Ptr<DirectedNode>> unknown_parents;

      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      // HERE
      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////
      ////////////////////////////////////////////////////

      for (const Ptr<MoralNode> &moral : host_->elements()) {
        Ptr<DirectedNode> directed = moral->base_node();
        for (const Ptr<DirectedNode> &parent : directed->parents()) {
          if (unique_parent_nodes.count(parent) == 0) {
            if (!parent_distribution->host()->contains(parent)) {
              report_error("Parent distribution is missing a parent node.");
            }
            if (parent_distribution->is_known(parent)) {
              known_parents.insert(parent);
            } else {
              unknown_parents.push_back(parent);
            }
            parent_nodes.push_back(parent);
            unique_parent_nodes.insert(parent);
          }
        }
      }

      // Get the marginal distribution of the unknown parents.
      Array parent_margin;
      if (parent_distribution) {
        parent_margin = parent_distribution->compute_margin(
            unknown_parents);
      }

      return 0.0;
    }

    Array CMD::compute_margin(const std::vector<Ptr<DirectedNode>> &subset) const {
      /////////////////////////////
      /////////////////////////////
      // TODO
      /////////////////////////////
      /////////////////////////////
      Array ans;
      return ans;
    }

  }  // namespace Graphical
}  // namespace BOOM

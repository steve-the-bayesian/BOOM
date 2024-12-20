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

            case NodeType::NUMERIC:
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
      } else if (directed->node_type() == NodeType::NUMERIC) {
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
    // Performs one step of the forward message passing algorithm.
    //
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

      // 1) Find all the host's nodes in the d-separator.  These have already
      //    been processed.  Separate them out into knowns vs unknowns, and find
      //    the marginal distribution of the unknowns.  If none of the nodes
      //    have parents in the parent clique the separator may be empty.
      NodeSet separator;
      NodeSetMarginalDistribution prior_margin(&separator);
      if (parent_distribution) {
        separator = parent_distribution->host()->intersection(*host_);
        prior_margin = parent_distribution->compute_margin(separator);
      }

      // This code can probably be sped up.  But the goal is to get it right
      // first.
      //
      // unknown_discrete_distribution_ needs to be set to p(unknowns | knowns)
      // = K * p(knowns | unknowns) * p(unknowns)
      //
      // 1) Set it to the prior... compute p(unknowns).
      // 2) Multiply by the likelihood.
      // 3) Normalize.
      //
      // To get this done we need to iterate over each value of the array.
      double total_prob = 0.0;

      // TODO: make sure this copy does not just copy pointers.  We don't want
      //       any data from 'data_point' getting overwritten.
      MixedMultivariateData scratch_data_point(data_point);

      for (auto it = unknown_discrete_distribution_.abegin();
           it != unknown_discrete_distribution_.aend();
           ++it) {
        const std::vector<int> &index(it.position());
        double prior = compute_prior_probability(index, prior_margin, scratch_data_point);
        double likelihood = compute_likelihood(index, data_point);
        double product = prior * likelihood;
        total_prob += product;
      }

      unknown_discrete_distribution_ /= total_prob;
      return total_prob;
    }

    // Compute the prior probability of a given cell in
    // unknown_discrete_distribution_.  Here "prior probability" means the
    // probability conditional on the preceding node in the junction tree.
    //
    // Args:
    //   index: The position (i, j, k, ...) in the
    //     unknown_discrete_distribution_ table.
    //   separator_margin: The marginal distribution of the nodes that appear in
    //     both this clique and the parent clique in the junction tree.
    //   scratch_data_point:  scratch workspace
    //
    // Details:
    //   This function works by writing the value of 'index' to
    //   'scratch_data_point', then evaluating the probability of
    //   'scratch_data_point' under the nodes in the margin.
    ///////////////////////
    ///////////////////////
    ///////////////////////
    // HERE
    ///////////////////////
    ///////////////////////
    ///////////////////////
    double NSMD::compute_prior_probability(
        const std::vector<int> &index,
        const NodeSetMarginalDistribution &separator_margin,
        MixedMultivariateData &scratch_data_point) const {
      double logprob = 0.0;

      // Need the set of nodes in the separator, and the set not in the
      // separator.
      //
      // For the requested index

      if (!separator_margin.host()->empty()) {
        // Need to take index and pull out the indices corresponding to the
        // variables in the margin.

        // We need a way of mapping between index positions and nodes in host().

        // TODO:
        std::vector<int> subset_index(index);
        logprob = log(separator_margin.unknown_discrete_distribution()[
            subset_index]);
        if (!std::isfinite(logprob)) {
          return 0.0;
        } else {

        }
      }

      std::vector<int> increment_index;

      for (const Ptr<Node> &node : unknown_discrete_nodes_) {
        if (separator_margin.host()->contains(node)) {

        }
      }

      return logprob;
    }

    double NSMD::compute_likelihood(
        const std::vector<int> &index,
        const MixedMultivariateData &data_point) const {


      report_error("compute_likelihood not yet implemented.");
      return negative_infinity();
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

            case NodeType::NUMERIC:
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

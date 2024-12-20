#ifndef BOOM_MODELS_GRAPHICAL_NODESET_MARGINAL_DISTRIBUTION_HPP_
#define BOOM_MODELS_GRAPHICAL_NODESET_MARGINAL_DISTRIBUTION_HPP_

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

#include "Models/Graphical/NodeSet.hpp"
#include "LinAlg/Array.hpp"

namespace BOOM {
  namespace Graphical {
    class NodeSetMarginalDistribution;
    void intrusive_ptr_add_ref(NodeSetMarginalDistribution *d);
    void intrusive_ptr_release(NodeSetMarginalDistribution *d);

    //==========================================================================
    // The marginal distribution for a set of nodes describing a mix of
    // categorical and conditionally Gaussian distributions.  Some values are
    // marked as known.
    //
    // When passed forward through a junction tree, this class gives the
    // marginal (joint) distribution of the variables in the node set given their
    // ancestors.
    //
    // When passed backward through a junction tree, a node set marginal
    // distribution is updated to contain the marginal distribution of the
    // node set variables given all available evidence.
    class NodeSetMarginalDistribution : private RefCounted {
      friend void intrusive_ptr_add_ref(NodeSetMarginalDistribution *d);
      friend void intrusive_ptr_release(NodeSetMarginalDistribution *d);

     public:
      // The NodeSetMarginalDistribution never owns the NodeSet it describes.
      NodeSetMarginalDistribution(const NodeSet *nodes);

      // Args:
      //   data_point:  The data point containing the evidence to be collected.
      //
      // Effects:
      //   - The following data structures are populated:
      //     * known_discrete_variables_
      //     * known_gaussian_variables_
      //     * unknown_discrete_nodes_
      //     * unknown_gaussian_nodes_
      void resize(const MixedMultivariateData &data_point);

      // Initialize the forward algorithm by computing the marginal distribution
      // of the variables in the node set.
      //
      // Args:
      //   data_point: The data point containing the evidence to be collected.
      //
      // Returns:
      //   The log density of the variables in data_point covered by
      //   this node set.
      //
      // Throws:
      //   An exception is generated if this node set contains nodes with parents
      //   outside the node set.
      double initialize_forward(const MixedMultivariateData &data_point) {
        return forward_increment(data_point, nullptr);
      }

      // Compute the marginal distribution of the variable in this node set
      // conditional on their ancestors in the directed graph.
      //
      // Args:
      //   data_point: The data point containing the evidence to be collected.
      //   parent: The preceding node set in the moral graph tree-of-node sets.
      //     If nullptr then this node set is the root of the node-set-tree.
      //
      // Returns:
      //   The log density of the observed but previously unprocessed variables
      //   in this node set.
      double forward_increment(const MixedMultivariateData &data_point,
                               const NodeSetMarginalDistribution *parent);

      // Compute the marginal distribution of the variables in 'subset.'
      //
      // Args:
      //   subset: A set of variables for which a marginal distribution is
      //     desired.  An exception is thrown if any variables in the subset are
      //     not covered by the current object.
      //
      // Returns:
      //   A distribution describing the requested subset.
      NodeSetMarginalDistribution compute_margin(const NodeSet &subset) const;

      // Returns true iff node is present in either known_gaussian_variables_ or
      // known_discrete_variables_.  These data structures are populated when
      // 'resize()' is called.
      bool is_known(const Ptr<Node> &node) const;

      // The variable number of this node in the subset.  0, 1, 2, etc.  If the
      // variable is not in the subset then return -1.
      int node_set_index(const Ptr<Node> &node) const;

      // Return the index of the supplied node in the
      // unknown_discrete_distribution_ array.  If 'node' is not an unknown
      // discrete variable then -1 is returned.
      int unknown_discrete_index(const Ptr<Node> &node) const {
        return unknown_discrete_nodes_.index(node);
      }

      // A pointer to the node set that this marginal distribution describes.
      const NodeSet *host() const {
        return host_;
      }

      const Array &unknown_discrete_distribution() const {
        return unknown_discrete_distribution_;
      }

      void set_unknown_discrete_distribution(const Array &dist) {
        unknown_discrete_distribution_ = dist;
      }

      const NodeSet &unknown_discrete_nodes() const {
        return unknown_discrete_nodes_;
      }

      const std::map<Ptr<Node>, int> &known_discrete_variables() const {
        return known_discrete_variables_;
      }

      // Compute the prior probability that the vector of
      // unknown_discrete_nodes_ is in the specified configuration.
      //
      // Args:
      //   index: The subset of variables for which the prior probability should
      //     be computed.
      //   separator_margin: A marginal distribution containing the distribution
      //     of any unknown parents in index.
      //   scratch_data_point: A workspace the prior can use when computing
      //     averages.
      double compute_prior_probability(
          const std::vector<int> &index,
          const NodeSetMarginalDistribution &separator_margin,
          MixedMultivariateData &scratch_data_point) const;

      // TODO: verify
      //
      // Computes the conditional likelihood of the given elements of the
      // supplied data_point.  The likelihood conditions on any parents in the
      // directed graph, and averages over any missing values.
      double compute_likelihood(
          const std::vector<int> &index,
          const MixedMultivariateData &data_point) const;

     private:
      const NodeSet *host_;

      // The values in 'data_point' associated with their respective nodes.
      std::map<Ptr<Node>, int> known_discrete_variables_;
      std::map<Ptr<Node>, double> known_gaussian_variables_;

      // The set of nodes in the node set for which the given data point contains
      // unknown (missing) values.  The number of discrete nodes in this vector
      // determines the number of dimensions in unknown_discrete_distribution_,
      // and their order corresponds to the dimensions of that array.
      NodeSet  unknown_discrete_nodes_;
      Array unknown_discrete_distribution_;

      std::vector<Ptr<Node>> unknown_gaussian_nodes_;
      Vector unknown_gaussian_potential_means_;
      SpdMatrix unknown_gaussian_potential_precisions_;
    };

  }
}

#endif  //  BOOM_MODELS_GRAPHICAL_NODESET_MARGINAL_DISTRIBUTION_HPP_

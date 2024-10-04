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
      friend void intrusive_ptr_add_ref(NodeSetMarginalDistribution *d) {
        d->up_count();
      }
      friend void intrusive_ptr_release(NodeSetMarginalDistribution *d) {
        d->down_count();
        if (d->ref_count() == 0) delete d;
      }

     public:
      NodeSetMarginalDistribution(NodeSet<Node> *nodes,
                                  bool assume_ownership = false);

      NodeSetMarginalDistribution(NodeSet<DirectedNode> *nodes);

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

      // Need a list of names to marginalize over.
      void marginalize();

      // Compute the marginal distribution of the set of (unknown) variables in subset, by
      NodeSetMarginalDistribution compute_margin(const NodeSet<Node> &subset) const;
      NodeSetMarginalDistribution compute_margin(const NodeSet<DirectedNode> &subset) const;

      // Returns true iff node is present in either known_gaussian_variables_ or
      // known_discrete_variables_.  These data structures are populated when
      // 'resize()' is called.
      bool is_known(const Ptr<DirectedNode> &node) const;

      // A pointer to the node set that this marginal distribution describes.
      const NodeSet<Node> *host() const {
        return host_;
      }

      NodeSet<DirectedNode> to_directed(const NodeSet<Node> &moral) const;

     private:

      NodeSet<Node> *host_;

      // For use with NodeSet's that own their host.
      Ptr<NodeSet<Node>> owned_host_maybe_null_dont_access_directly_;

      // The values in 'data_point' associated with their respective nodes.
      std::map<Ptr<DirectedNode>, int> known_discrete_variables_;
      std::map<Ptr<DirectedNode>, double> known_gaussian_variables_;

      // The set of nodes in the node set for which the given data point contains
      // unknown (missing) values.  The number of discrete nodes in this vector
      // determines the number of dimensions in unknown_discrete_distribution_,
      // and their order corresponds to the dimensions of that array.
      //
      // The use of std::vector here instead of some other data structure
      // (e.g. NodeSet) is purposeful, because the order of the nodes matters.
      std::vector<Ptr<DirectedNode>>  unknown_discrete_nodes_;
      Array unknown_discrete_distribution_;

      std::vector<Ptr<DirectedNode>> unknown_gaussian_nodes_;
      Vector unknown_gaussian_potential_means_;
      SpdMatrix unknown_gaussian_potential_precisions_;
    };

  }
}

#endif  //  BOOM_MODELS_GRAPHICAL_NODESET_MARGINAL_DISTRIBUTION_HPP_

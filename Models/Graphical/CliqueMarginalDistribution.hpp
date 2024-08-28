#ifndef BOOM_MODELS_GRAPHICAL_CLIQUE_MARGINAL_DISTRIBUTION_HPP_
#define BOOM_MODELS_GRAPHICAL_CLIQUE_MARGINAL_DISTRIBUTION_HPP_

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

#include "Models/Graphical/Clique.hpp"
#include "LinAlg/Array.hpp"

namespace BOOM {
  namespace Graphical {
    //==========================================================================
    // A mix of categorical and conditionally Gaussian distributions.
    // Some values are marked as known.
    //
    // When passed forward through a junction tree, a clique marginal
    // distribution gives the marginal (joint) distribution of the variables in
    // the clique given their ancestors.
    //
    // When passed backward through a junction tree, a clique marginal
    // distribution is updated to contain the marginal distribution of the
    // clique variables given all available evidence.
    class CliqueMarginalDistribution : private RefCounted {
      friend void intrusive_ptr_add_ref(CliqueMarginalDistribution *d) {
        d->up_count();
      }
      friend void intrusive_ptr_release(CliqueMarginalDistribution *d) {
        d->down_count();
        if (d->ref_count() == 0) delete d;
      }

     public:
      CliqueMarginalDistribution(Clique *clique);

      // Store values for the known nodes.  Resize
      void resize(const MixedMultivariateData &data_point);

      // Initialize the forward algorithm by computing the marginal distribution
      // of the variables in the clique.
      //
      // Args:
      //   data_point: The data point containing the evidence to be collected.
      //
      // Returns:
      //   The log density of the variables in data_point covered by
      //   this clique.
      //
      // Throws:
      //   An exception is generated if this clique contains nodes with parents
      //   outside the clique.
      double initialize_forward(const MixedMultivariateData &data_point);

      // Compute the marginal distribution of the variable in this clique
      // conditional on their ancestors in the directed graph.
      //
      // Args:
      //   data_point:  The data point containing the evidence to be collected.
      //   parent: The preceding clique in the moral graph tree-of-cliques.  If
      //     nullptr then this clique is the root of the clique-tree.
      //
      // Returns:
      //   The log density of the variables in this clique not described by a
      //   parent.
      double forward_increment(const MixedMultivariateData &data_point,
                               const CliqueMarginalDistribution *parent);

      // Need a list of names to marginalize over.
      void marginalize();

      Array compute_margin(const std::vector<Ptr<DirectedNode>> &subset) const;

      bool is_known(const Ptr<DirectedNode> &node) const;

      const Clique *host() const {
        return host_;
      }

     private:
      Clique *host_;

      std::map<Ptr<DirectedNode>, int> known_discrete_variables_;
      std::map<Ptr<DirectedNode>, double> known_gaussian_variables_;

      std::vector<Ptr<DirectedNode>>  unknown_nodes_;

      Array unknown_discrete_distribution_;
      Vector unknown_gaussian_potential_means_;
      SpdMatrix unknown_gaussian_potential_precisions_;
    };

  }
}

#endif  //  BOOM_MODELS_GRAPHICAL_CLIQUE_MARGINAL_DISTRIBUTION_HPP_

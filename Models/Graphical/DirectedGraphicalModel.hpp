#ifndef BOOM_MODELS_GRAPHICAL_DIRECTEDGRAPHICALMODEL_HPP_
#define BOOM_MODELS_GRAPHICAL_DIRECTEDGRAPHICALMODEL_HPP_

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

#include "Models/ModelTypes.hpp"
#include "Models/Policies/MultivariateDataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"

#include "stats/DataTable.hpp"  // home of MixedMultivariateData

#include "Models/Graphical/Node.hpp"
#include "Models/Graphical/Clique.hpp"
#include "Models/Graphical/JunctionTree.hpp"

#include "cpputil/SortedVector.hpp"
#include <vector>

namespace BOOM {

  // A "directed graphical model" (DGM) is also known by other names, including
  // a "Bayes net".  Each node in a DGM represents a variable in a data frame.
  // The graph is "directed" in that nodes have parents and children, with
  // arrows indicating one-way conditional independence.  A node is
  // conditionally independent of all other ancestors given its parents.
  //
  // The most important algorithm in a DGM is the message passing algorithm, by
  // which one can work with the conditional distribution of missing variables
  // given observed ones.  The message passing algorithm is the "graph" version
  // of the forward-backward algorithm for hidden Markov models, or the Kalman
  // filter/smoother algorithms for state space models.  For these algorithms to
  // work, the variables in question must either be discrete (with finite state
  // space) or conditionally Gaussian (CG).  However, one can often pair more
  // general DGM's with latent variables, conditional on which the discrete/CG
  // assumption is satisfied.
  class DirectedGraphicalModel
      : public CompositeParamPolicy,
        public MultivariateDataPolicy,
        public PriorPolicy
  {
   public:
    using Node = ::BOOM::Graphical::Node;

    DirectedGraphicalModel();

    // Add a node to the graph.  The links between this node and its parents and
    // children must be managed separately.
    void add_node(const Ptr<Node> &node);

    // Return the log density of the given data_point.  Some elements of
    // data_point may be missing.
    double logp(const MixedMultivariateData &data_point) const;

    void set_triangulation_heuristic(
        const Graphical::JunctionTree::Criterion &criterion) {
      junction_tree_.set_triangulation_heuristic(criterion);
    }

    // Fill any missing values in data_point with values imputed from their
    // joint posterior distribution.
    //
    // Args:
    //   data_point: The observation with some elements of missing data to be
    //     imputed.
    //   rng: Then random number generator used as a source of randomness for
    //     the imputation.
    void impute_missing_values(MixedMultivariateData &data_point, RNG &rng) {
      ensure_junction_tree();
      junction_tree_.impute_missing_values(data_point, rng);
    }

   private:
    // If the junction tree has not yet been built, or if something has been
    // done to invalidate it, rebuild the tree.
    //
    // Effects:
    //   If junction_tree_current_ is false then junction_tree_ is rebuilt.
    //   junction_tree_current_ is set to true.
    void ensure_junction_tree() const;

    // Order nodes by their ID.
    struct IdLess {
      bool operator()(const Ptr<::BOOM::Graphical::Node> &n1,
                      const Ptr<::BOOM::Graphical::Node> &n2) const {
        return n1->id() < n2->id();
      }
    };

    SortedVector<Ptr<Graphical::Node>, IdLess> nodes_;
    mutable bool junction_tree_current_;
    mutable Graphical::JunctionTree junction_tree_;
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_GRAPHICAL_DIRECTEDGRAPHICALMODEL_HPP_

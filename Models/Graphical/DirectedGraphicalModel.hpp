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

#include "cpputil/SortedVector.hpp"

namespace BOOM {

  class DirectedGraphicalModel
      : public CompositeParamPolicy,
        public MultivariateDataPolicy,
        public PriorPolicy
  {
    using Node = ::BOOM::Graphical::Node;
    using DirectedNode = ::BOOM::Graphical::DirectedNode;
    using MoralNode = ::BOOM::Graphical::MoralNode;
    using Clique = ::BOOM::Graphical::Clique;

   public:
    DirectedGraphicalModel();

    void add_node(const Ptr<DirectedNode> &node);

    double logp(const MixedMultivariateData &data_point) const;

    // Create a
    std::vector<Ptr<MoralNode>> create_moral_graph(
        const std::vector<Ptr<DirectedNode>> &nodes) const;

    void accumulate_evidence(const Ptr<MixedMultivariateData> &data_point);
    void distribute_evidence(const Ptr<MixedMultivariateData> &data_point);

   private:

    // If the junction tree has not yet been built, or if something has been
    // done to invalidate it, rebuild the tree.
    //
    // Effects:
    //   junction_tree_ is rebuilt if needed.
    //   junction_tree_current_ is set to true.
    void ensure_junction_tree() const;

    // A class to represent the desired node ordering, which is to compare two
    // nodes by their id.
    struct IdLess {
      bool operator()(const Ptr<::BOOM::Graphical::Node> &n1,
                      const Ptr<::BOOM::Graphical::Node> &n2) const {
        return n1->id() < n2->id();
      }
    };

    SortedVector<Ptr<DirectedNode>, IdLess> nodes_;

    mutable bool junction_tree_current_;
    mutable std::vector<Ptr<Clique>> junction_tree_;

    // When building a junction tree, the moral graph needs to be triangulated,
    // which means nodes need to be added
    std::function<double(Ptr<MoralNode>)> triangulation_heuristic_;
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_GRAPHICAL_DIRECTEDGRAPHICALMODEL_HPP_

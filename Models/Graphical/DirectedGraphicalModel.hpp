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

    void ensure_junction_tree() const;


    // Compare two nodes by their id.
    struct IdLess {
      bool operator()(const Ptr<::BOOM::Graphical::Node> &n1,
                      const Ptr<::BOOM::Graphical::Node> &n2) const {
        return n1->id() < n2->id();
      }
    };

    SortedVector<Ptr<DirectedNode>,
                 IdLess> nodes_;

    mutable bool junction_tree_current_;
    mutable std::vector<Ptr<Clique>> junction_tree_;

    std::function<double(Ptr<MoralNode>)> triangulation_heuristic_;
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_GRAPHICAL_DIRECTEDGRAPHICALMODEL_HPP_

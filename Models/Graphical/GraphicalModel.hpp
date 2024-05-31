#ifndef BOOM_MODELS_GRAPHICAL_GRAPHICALMODEL_HPP_
#define BOOM_MODELS_GRAPHICAL_GRAPHICALMODEL_HPP_

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
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"

#include "stats/DataTable.hpp"  // home of MixedMultivariateData

#include "Models/GraphicalModel/Node.hpp"

#include <vector>
#include <map>

namespace BOOM {

  // A graphical model is a model for multivariate data of mixed type.  The
  // graphical models handled here are directed graphical models also known as
  // "Bayesian networks" (See https://en.wikipedia.org/wiki/Bayesian_network).
  //
  // The model is described by a collection of nodes, corresponding to varibles
  // in a data frame.  The nodes are connected by directed edges where the
  // direction of the connection refers to which nodes are 
  class GraphicalModel
      : public CompositeParamPolicy,
        public IID_DataPolicy<MixedMultivariateData>,
        public PriorPolicy
  {
    using Graphical::Node;

   public:
    void add_node(const Ptr<Node> &node);

    // If each element of data_point is fully observed, logp returns the sum of
    // the result of calling logp on each of the nodes applied to its portion of
    // data_point.
    double logp(const MixedMultivariateData &data_point) const;

    // Simulate a new data point.
    //
    // Args:
    //   rng: The random number generator used to drive the simulation.
    //   input:  A data point with data elements
    Ptr<MixedMultivariateData> simulate(RNG &rng, Ptr<MixedMultivariateData> &input);
    
   private:
    std::vector<Ptr<Node>> nodes_;
    std::map<std::string, Ptr<Node>> node_names_;
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_GRAPHICAL_GRAPHICALMODEL_HPP_

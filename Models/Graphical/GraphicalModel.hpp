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

namespace BOOM {

  class GraphicalModel
      : public CompositeParamPolicy,
        public IID_DataPolicy<MixedMultivariateData>,
        public PriorPolicy
  {
    using Graphical::Node;

   public:
    double logp(const MixedMultivariateData &data_point) const;

    void add_data(const Ptr<DataTable> &data_table);
    void add_data(const Ptr<MixedMultivariateData> &data_point);

   private:
    std::vector<Ptr<DirectedNode>> nodes_;
  };

}  // namespace BOOM

#endif  //  BOOM_MODELS_GRAPHICAL_GRAPHICALMODEL_HPP_

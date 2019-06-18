/*
  Copyright (C) 2019 Steven L. Scott

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

#include "Models/Glm/IndependentRegressionModels.hpp"

namespace BOOM {

  IndependentRegressionModels::IndependentRegressionModels(int xdim, int ydim)
  {
    models_.reserve(ydim);
    for (int i = 0; i < ydim; ++i) {
      NEW(RegressionModel, model)(xdim);
      ParamPolicy::add_model(model);
      models_.push_back(model);
    }
  }

  IndependentRegressionModels::IndependentRegressionModels(
      const IndependentRegressionModels &rhs)
      : Model(rhs),
        CompositeParamPolicy(rhs),
        NullDataPolicy(rhs),
        PriorPolicy(rhs)
  {
    models_.reserve(rhs.ydim());
    for (int i = 0; i < rhs.models_.size(); ++i) {
      models_.push_back(rhs.models_[i]->clone());
      ParamPolicy::add_model(models_.back());
    }
  }

  void IndependentRegressionModels::clear_data() {
    DataPolicy::clear_data();
    for (int i = 0; i < models_.size(); ++i) {
      models_[i]->clear_data();
    }
  }
  
}  // namespace BOOM

#ifndef BOOM_GLM_INDEPENDENT_REGRESSION_MODELS_HPP_
#define BOOM_GLM_INDEPENDENT_REGRESSION_MODELS_HPP_
/*
  Copyright (C) 2005-2019 Steven L. Scott

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

#include "Models/Glm/RegressionModel.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A "multivariate regression" formed by a sequence of independent scalar
  // regression models.  The models must all have the same predictor dimension,
  // but are otherwise unconstrained.
  class IndependentRegressionModels
      : public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy {
   public:
    IndependentRegressionModels(int xdim, int ydim);
    IndependentRegressionModels(const IndependentRegressionModels &rhs);
    
    IndependentRegressionModels *clone() const override {
      return new IndependentRegressionModels(*this);
    }
    
    int xdim() const {return models_[0]->xdim();}
    int ydim() const {return models_.size();}

    void clear_data() override;
    
    Ptr<RegressionModel> model(int i) {return models_[i];}
    const Ptr<RegressionModel> model(int i) const {return models_[i];}
    
   private:
    std::vector<Ptr<RegressionModel>> models_;
  };

}  // namespace BOOM

#endif  // BOOM_GLM_INDEPENDENT_REGRESSION_MODELS_HPP_

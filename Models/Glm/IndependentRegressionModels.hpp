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

  //===========================================================================
  // A "multivariate GLM" formed by a sequence of independent generalized linear
  // models.  The models must all have the same predictor dimension, but are
  // otherwise unconstrained.
  template <class GLM>
  class IndependentGlms
      : public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy,
        public PosteriorModeModel {
   public:
    IndependentGlms(int xdim, int ydim)
    {
      models_.reserve(ydim);
      for (int i = 0; i < ydim; ++i) {
        NEW(GLM, model)(xdim);
        ParamPolicy::add_model(model);
        models_.push_back(model);
      }
    }

    IndependentGlms(const IndependentGlms &rhs)
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

    IndependentGlms * clone() const override {
      return new IndependentGlms(*this);
    }

    int xdim() const {return models_.empty() ? 0 : models_[0]->xdim();}
    int ydim() const {return models_.size();}

    void clear_data() override {
      DataPolicy::clear_data();
      for (auto &el : models_) {
        el->clear_data();
      }
    }

    GLM *model(int i) { return models_[i].get(); }
    const GLM *model(int i) const { return models_[i].get(); }

   private:
    std::vector<Ptr<GLM>> models_;
  };

  // IndependentRegressionModels needs to be kept here for legacy reasons.
  using IndependentRegressionModels = IndependentGlms<RegressionModel>;

}  // namespace BOOM

#endif  // BOOM_GLM_INDEPENDENT_REGRESSION_MODELS_HPP_

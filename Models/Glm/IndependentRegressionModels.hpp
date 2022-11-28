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
#include "Models/Glm/TRegression.hpp"
#include "Models/Policies/NullDataPolicy.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // An abstract base class for a "multivariate" generalized linear model formed
  // by stacking 'k' independent GLM's.
  class IndependentGlms
      : public PosteriorModeModel {
   public:
    virtual int xdim() const = 0;
    virtual int ydim() const = 0;
    virtual PosteriorModeModel *model(int which) = 0;
    virtual const PosteriorModeModel *model(int which) const = 0;

   protected:
    void clear_client_data();
  };

  // A "multivariate regression" formed by a sequence of independent scalar
  // regression models.  The models must all have the same predictor dimension,
  // but are otherwise unconstrained.
  class IndependentRegressionModels
      : public IndependentGlms,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy {
   public:
    IndependentRegressionModels(int xdim, int ydim);
    IndependentRegressionModels(const IndependentRegressionModels &rhs);

    IndependentRegressionModels *clone() const override {
      return new IndependentRegressionModels(*this);
    }

    int xdim() const override {return models_[0]->xdim();}
    int ydim() const override {return models_.size();}

    void clear_data() override;

    RegressionModel * model(int i) override {return models_[i].get();}
    const RegressionModel * model(int i) const override {
      return models_[i].get();
    }

   private:
    std::vector<Ptr<RegressionModel>> models_;
  };

  //===========================================================================
  // A "multivariate Student T" regression formed by stacking independent scalar
  // Student T regressions.
  class IndependentStudentRegressionModels
      : public IndependentGlms,
        public CompositeParamPolicy,
        public NullDataPolicy,
        public PriorPolicy {
   public:
    IndependentStudentRegressionModels(int xdim, int ydim);
    IndependentStudentRegressionModels(
        const IndependentStudentRegressionModels &rhs);
    IndependentStudentRegressionModels * clone() const override;

    int xdim() const override {return models_.empty() ? 0 : models_[0]->xdim();}
    int ydim() const override {return models_.size();}
    void clear_data() override;

    TRegressionModel *model(int i) override {
      return models_[i].get();
    }

    const TRegressionModel *model(int i) const override {
      return models_[i].get();
    }

   private:
    std::vector<Ptr<TRegressionModel>> models_;
  };

}  // namespace BOOM

#endif  // BOOM_GLM_INDEPENDENT_REGRESSION_MODELS_HPP_

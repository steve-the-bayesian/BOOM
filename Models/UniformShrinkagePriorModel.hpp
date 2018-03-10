// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#ifndef BOOM_UNIFORM_SHRINKAGE_PRIOR_MODEL_HPP_
#define BOOM_UNIFORM_SHRINKAGE_PRIOR_MODEL_HPP_

#include "Models/DoubleModel.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A one-parameter model, with parameter z0 representing the median,
  // and (properly normalized) density    f(alpha) = z0 / (z0 + alpha)^2
  // with support on alpha > 0.
  // See Christiansen and Morris (1997, JASA, Hierarchical Poisson
  // Regression Modeling).
  class UniformShrinkagePriorModel : public ParamPolicy_1<UnivParams>,
                                     public IID_DataPolicy<DoubleData>,
                                     public PriorPolicy,
                                     public DiffDoubleModel,
                                     public NumOptModel {
   public:
    explicit UniformShrinkagePriorModel(double median = 1.0);
    UniformShrinkagePriorModel *clone() const override;

    void set_median(double z0);
    double median() const { return prm_ref().value(); }

    double Logp(double x, double &g, double &h, uint nd) const override;
    double Loglike(const Vector &z0, Vector &gradient, Matrix &Hessian,
                   uint nd) const override;

    double sim(RNG &rng = GlobalRng::rng) const override;
    int number_of_observations() const override { return dat().size(); }
  };

}  // namespace BOOM

#endif  //  BOOM_UNIFORM_SHRINKAGE_PRIOR_MODEL_HPP_

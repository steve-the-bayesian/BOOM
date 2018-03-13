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

#ifndef BOOM_QUANTILE_REGRESSION_MODEL_HPP_
#define BOOM_QUANTILE_REGRESSION_MODEL_HPP_

#include "LinAlg/Vector.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A quantile regression model posits that a specific quantile of a
  // random variable (e.g. the median, the 90th percentile, ...) is a
  // linear function of predictors X.  There is no notion of residual
  // variance in a QR model.
  //
  // The model is typically fit using the observation that the median
  // is the optimal action under L1 loss, so median regression
  // minimizes \sum |y_i - beta.dot(x_i)|.  Other quantiles r (in (0,
  // 1)) replace absolute value with \sum rho_p(y_i - beta.dot(x_i)),
  // where rho_p(u) = u * (p - I(u < 0)).  This is the "check
  // function" which has slope p for u > 0 and slope p-1 for u < 0.
  //
  // The pseudo-likelihood for this function exponentiates the
  // negative of the loss function sum_i -rho_p(y_i - beta.dot(x_i)).
  class QuantileRegressionModel : public GlmModel,
                                  public ParamPolicy_1<GlmCoefs>,
                                  public IID_DataPolicy<RegressionData>,
                                  public PriorPolicy {
   public:
    explicit QuantileRegressionModel(uint beta_dim, double quantile = .5,
                                     bool include_all = true);
    explicit QuantileRegressionModel(const Vector &beta, double quantile = .5);
    QuantileRegressionModel *clone() const override;

    GlmCoefs &coef() override { return ParamPolicy::prm_ref(); }
    const GlmCoefs &coef() const override { return ParamPolicy::prm_ref(); }
    Ptr<GlmCoefs> coef_prm() override { return ParamPolicy::prm(); }
    const Ptr<GlmCoefs> coef_prm() const override { return ParamPolicy::prm(); }

    double quantile() const { return quantile_; }

   private:
    double quantile_;
  };

}  // namespace BOOM

#endif  //  BOOM_QUANTILE_REGRESSION_MODEL_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#ifndef PROBIT_REGRESSION_HPP
#define PROBIT_REGRESSION_HPP

#include "uint.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Policies/CompositeParamPolicy.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "TargetFun/TargetFun.hpp"
#include "numopt.hpp"

namespace BOOM {

  class ProbitRegressionModel : public GlmModel,
                                public NumOptModel,
                                public ParamPolicy_1<GlmCoefs>,
                                public IID_DataPolicy<BinaryRegressionData>,
                                public PriorPolicy {
   public:
    explicit ProbitRegressionModel(const Vector &beta);
    ProbitRegressionModel(const Matrix &X, const Vector &y);
    ProbitRegressionModel(const ProbitRegressionModel &);
    ProbitRegressionModel *clone() const override;

    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;

    virtual double pdf(const Ptr<Data> &, bool) const;
    virtual double pdf(const Ptr<BinaryRegressionData> &, bool) const;
    virtual double pdf(bool y, const Vector &x, bool logscale) const;

    // The dimension here and in log_likelihood is the number of
    // included variables.
    double Loglike(const Vector &beta, Vector &g, Matrix &h,
                   uint nd) const override;

    // call with *g=0 if you don't want any derivatives.  Call with
    // *g!=0 and *h=0 if you only want first derivatives.
    // if(initialize_derivs) then *g and *h will be set to zero.
    // Otherwise they will be incremented.
    double log_likelihood(const Vector &beta, Vector *g, Matrix *h,
                          bool initialize_derivs = true) const;
    using LoglikeModel::log_likelihood;
    d2TargetFunPointerAdapter log_likelihood_tf() const;

    bool sim(const Vector &x, RNG &rng = GlobalRng::rng) const;
    Ptr<BinaryRegressionData> sim(RNG &rng = GlobalRng::rng) const;
  };

}  // namespace BOOM

#endif  // PROBIT_REGRESSION_HPP

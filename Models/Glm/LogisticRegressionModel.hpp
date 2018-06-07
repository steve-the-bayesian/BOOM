// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include "uint.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "TargetFun/TargetFun.hpp"
#include "numopt.hpp"

namespace BOOM {

  class LogisticRegressionModel : public GlmModel,
                                  public NumOptModel,
                                  virtual public MixtureComponent,
                                  public ParamPolicy_1<GlmCoefs>,
                                  public IID_DataPolicy<BinaryRegressionData>,
                                  public PriorPolicy {
   public:
    explicit LogisticRegressionModel(uint beta_dim, bool include_all = true);
    explicit LogisticRegressionModel(const Vector &beta);
    LogisticRegressionModel(const Matrix &X, const Vector &y, bool add_int);
    LogisticRegressionModel(const LogisticRegressionModel &);
    LogisticRegressionModel *clone() const override;

    GlmCoefs &coef() override { return ParamPolicy::prm_ref(); }
    const GlmCoefs &coef() const override { return ParamPolicy::prm_ref(); }
    Ptr<GlmCoefs> coef_prm() override { return ParamPolicy::prm(); }
    const Ptr<GlmCoefs> coef_prm() const override { return ParamPolicy::prm(); }

    virtual double pdf(const Ptr<Data> &dp, bool logscale) const;
    double pdf(const Data *dp, bool logscale) const override;
    double logp(bool y, const Vector &x) const;
    int number_of_observations() const override { return dat().size(); }

    // In the following, 'beta' refers to the set of nonzero
    // "included" coefficients, so its dimension might be less than
    // the number of columns in the design matrix.
    double Loglike(const Vector &beta, Vector &g, Matrix &h,
                   uint nd) const override;
    virtual double log_likelihood(const Vector &beta, Vector *g, Matrix *h,
                                  bool initialize_derivs = true) const;
    using LoglikeModel::log_likelihood;
    d2TargetFunPointerAdapter log_likelihood_tf() const;

    virtual SpdMatrix xtx() const;

    // when modeling rare events it can be convenient to retain all
    // the events and 100 * alpha% of the non-events.
    void set_nonevent_sampling_prob(double alpha);
    double log_alpha() const;

   private:
    double log_alpha_;  // alpha is the probability that a 'zero'
                        // (non-event) is retained in the data.  It is
                        // assumed that the data retains all the
                        // events and 100 alpha% of the non-events
  };

}  // namespace BOOM

#endif  // LOGISTIC_REGRESSION_HPP

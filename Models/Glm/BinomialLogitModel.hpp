// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_BINOMIAL_LOGIT_MODEL_HPP_
#define BOOM_BINOMIAL_LOGIT_MODEL_HPP_
#include "uint.hpp"
#include "Models/EmMixtureComponent.hpp"
#include "Models/Glm/BinomialRegressionData.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "TargetFun/TargetFun.hpp"
#include "numopt.hpp"

namespace BOOM {
  // Logistic regression model with binomial (binned) training data.
  class BinomialLogitModel : public GlmModel,
                             public NumOptModel,
                             public ParamPolicy_1<GlmCoefs>,
                             public IID_DataPolicy<BinomialRegressionData>,
                             public PriorPolicy,
                             virtual public MixtureComponent {
   public:
    explicit BinomialLogitModel(uint beta_dim, bool include_all = true);
    explicit BinomialLogitModel(const Vector &beta);

    // Use this constructor if the model needs to share its
    // coefficient vector with another model.
    explicit BinomialLogitModel(const Ptr<GlmCoefs> &beta);

    BinomialLogitModel(const Matrix &X, const Vector &y, const Vector &n);
    BinomialLogitModel(const BinomialLogitModel &);
    BinomialLogitModel *clone() const override;

    GlmCoefs &coef() override { return ParamPolicy::prm_ref(); }
    const GlmCoefs &coef() const override { return ParamPolicy::prm_ref(); }
    Ptr<GlmCoefs> coef_prm() override { return ParamPolicy::prm(); }
    const Ptr<GlmCoefs> coef_prm() const override { return ParamPolicy::prm(); }

    double success_probability(const Vector &x) const;
    double success_probability(const VectorView &x) const;
    double success_probability(const ConstVectorView &x) const;
    double failure_probability(const Vector &x) const;
    double failure_probability(const VectorView &x) const;
    double failure_probability(const ConstVectorView &x) const;

    double pdf(const Data *dp, bool logscale) const override;
    virtual double pdf(const BinomialRegressionData *dp, bool logscale) const;
    virtual double logp(double y, double n, const Vector &x,
                        bool logscale) const;
    virtual double logp_1(bool y, const Vector &x, bool logscale) const;
    int number_of_observations() const override { return dat().size(); }

    // In the following, beta refers to the set of nonzero "included"
    // coefficients.
    double Loglike(const Vector &beta, Vector &g, Matrix &h,
                   uint nd) const override;
    virtual double log_likelihood(const Vector &beta, Vector *g, Matrix *h,
                                  bool initialize_derivs = true) const;
    using LoglikeModel::log_likelihood;
    d2TargetFunPointerAdapter log_likelihood_tf() const;

    virtual SpdMatrix xtx() const;

    // see comments in LogisticRegressionModel
    void set_nonevent_sampling_prob(double alpha);
    double log_alpha() const;

   private:
    double log_alpha_;  // see comments in logistic_regression_model
  };

}  // namespace BOOM

#endif  // BOOM_BINOMIAL_LOGIT_MODEL_HPP_

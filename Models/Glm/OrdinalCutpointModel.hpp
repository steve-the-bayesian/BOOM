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

#ifndef ORDINAL_CUTPOINT_MODEL_HPP
#define ORDINAL_CUTPOINT_MODEL_HPP

#include "Models/CategoricalData.hpp"
#include "Models/Glm/Glm.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "TargetFun/TargetFun.hpp"

// Model:  Y can be 0... M-1
// Pr(Y-m) = F(d[m]-btx) - F(d[m-1] - btx)
// where F is the link function and btx is "beta transpose x"
// d[] is the set of cutpoints with identifiability constraints:
// d[-1] = -infinity, d[0] = 0, d[M-1]=infinity

namespace BOOM {

  class OrdinalCutpointModel;

  class OrdinalCutpointBetaLogLikelihood : public TargetFun {
   public:
    explicit OrdinalCutpointBetaLogLikelihood(const OrdinalCutpointModel *m_);
    double operator()(const Vector &beta) const override;

   private:
    const OrdinalCutpointModel *m_;
  };

  class OrdinalCutpointDeltaLogLikelihood : public TargetFun {
   public:
    explicit OrdinalCutpointDeltaLogLikelihood(const OrdinalCutpointModel *m_);
    double operator()(const Vector &delta) const override;

   private:
    const OrdinalCutpointModel *m_;
  };

  class OrdinalCutpointModel : public ParamPolicy_2<GlmCoefs, VectorParams>,
                               public IID_DataPolicy<OrdinalRegressionData>,
                               public PriorPolicy,
                               public GlmModel,
                               public NumOptModel {
   public:
    OrdinalCutpointModel(const Vector &beta, const Vector &delta);
    OrdinalCutpointModel(const Vector &beta, const Selector &Inc,
                         const Vector &delta);
    OrdinalCutpointModel(const Selector &Inc, uint Maxscore);
    OrdinalCutpointModel(const Matrix &X, const Vector &y);
    OrdinalCutpointModel(const OrdinalCutpointModel &rhs);

    OrdinalCutpointModel *clone() const override = 0;

    // link_inv(eta) = probability
    // link(prob) = eta
    virtual double link_inv(double) const = 0;   // logit or probit
    virtual double dlink_inv(double) const = 0;  // derivative of link_inv

    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;

    Ptr<VectorParams> Delta_prm();
    const Ptr<VectorParams> Delta_prm() const;

    // inherits [Bb]eta()/set_[Bb]eta() from GlmModel
    double delta(uint) const;  // delta[0] = - infinity, delta[1] = 0
    const Vector &delta() const;
    void set_delta(const Vector &d);

    // Check to see if Delta satisfies constraint.
    bool check_delta(const Vector &Delta) const;

    // Args:
    //   beta_delta: A vector with leading elements corresponding to
    //     the set of nonzero "included" regression coefficients.  The
    //     remaining coefficients correspond to the vector of
    //     cutpoints 'delta'.
    //   g: Gradient vector (unused if nd == 0).  Dimension must match
    //     beta_delta.
    //   h: Hessian matrix (unused if nd < 2).  Dimension must match
    //     beta_delta.
    //   nd:  The number of derivatives desired.
    double Loglike(const Vector &beta_delta, Vector &g, Matrix &h,
                   uint nd) const override;
    double log_likelihood(const Vector &beta, const Vector &delta) const;
    using LoglikeModel::log_likelihood;
    OrdinalCutpointBetaLogLikelihood beta_log_likelihood() const;
    OrdinalCutpointDeltaLogLikelihood delta_log_likelihood() const;

    void initialize_params() override;
    void initialize_params(const Vector &counts);

    Vector CDF(const Vector &x) const;  // Pr(Y<y)

    virtual double pdf(const Ptr<Data> &, bool) const;
    double pdf(const Ptr<OrdinalRegressionData> &, bool) const;
    double pdf(const OrdinalData &y, const Vector &x, bool logscale) const;
    double pdf(uint y, const Vector &x, bool logscale) const;

    uint maxscore() const;  // maximum possible score allowed

    Ptr<OrdinalRegressionData> sim(RNG &rng = GlobalRng::rng);

   private:
    // interface is complicated
    double bd_loglike(Vector &gbeta, Vector &gdelta, Matrix &Hbeta,
                      Matrix &Hdelta, Matrix &Hbd, uint nd, bool b_derivs,
                      bool d_derivs) const;
    Ptr<CatKeyBase> simulation_key_;

    // Simulate latent variable from the link distribution.
    virtual double simulate_latent_variable(
        RNG &rng = GlobalRng::rng) const = 0;
  };

}  // namespace BOOM

#endif  // ORDINAL_CUTPOINT_MODEL_HPP

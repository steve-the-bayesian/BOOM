// Copyright (C) 2019 Steven L. Scott

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

#include "distributions.hpp"

namespace BOOM {

  class OrdinalCutpointModel;

  //===========================================================================
  // A target function for evaluating the conditional log likelihood of beta
  // given cutpoints.
  class OrdinalCutpointBetaLogLikelihood : public TargetFun {
   public:
    explicit OrdinalCutpointBetaLogLikelihood(const OrdinalCutpointModel *m_);
    double operator()(const Vector &beta) const override;

   private:
    const OrdinalCutpointModel *m_;
  };

  //===========================================================================
  // A target function for evaluating the conditional log likelihood of
  // cutpoints given beta.
  class OrdinalCutpointLogLikelihood : public TargetFun {
   public:
    explicit OrdinalCutpointLogLikelihood(const OrdinalCutpointModel *m_);
    double operator()(const Vector &minimal_cutpoints) const override;

   private:
    const OrdinalCutpointModel *m_;
  };

  //===========================================================================
  // An ordinal cutpoint model is a generalized linear model for ordinal data.
  // THe response Y can assume values in {0... M-1}.
  //
  // The model is Pr(Y == m) = F(d[m] - eta) - F(d[m-1] - eta) where F is the link
  // function (a cumulative distribution function, often the logistic function,
  // or normal CDF) and eta is "linear predictor".
  //
  // The notional vector d[] contains the set of cutpoints with identifiability
  // constraints: d[-1] = -infinity, d[0] = 0, d[M-1]=infinity, with d[i] < d[i+1]
  // otherwise.
  class OrdinalCutpointModel : public ParamPolicy_2<GlmCoefs, VectorParams>,
                               public IID_DataPolicy<OrdinalRegressionData>,
                               public PriorPolicy,
                               public GlmModel,
                               public NumOptModel,
                               public MixtureComponent {
   public:
    // Initialize parameters to be 
    OrdinalCutpointModel(int xdim, int nlevels);
    OrdinalCutpointModel(const Vector &beta, const Vector &cutpoints);
    OrdinalCutpointModel(const Vector &beta, const Selector &Inc,
                         const Vector &cutpoints);
    OrdinalCutpointModel(const Selector &Inc, uint Maxscore);
    OrdinalCutpointModel(const Matrix &X, const Vector &y);
    OrdinalCutpointModel(const OrdinalCutpointModel &rhs);
    
    OrdinalCutpointModel(OrdinalCutpointModel &&rhs) = default;

    OrdinalCutpointModel *clone() const override = 0;

    int number_of_observations() const override { return dat().size(); }

    // The link function, mapping a probaility to the real line.
    virtual double link(double p) const = 0;
    
    // The inverse of the link function.
    // Args:
    //   eta: The value of the random variable's mean on the unconstrained
    //     'linear predictor' scale.
    // 
    // Returns: The value of the random variable's mean on the [0, 1]
    //   probability scale.
    virtual double link_inv(double eta) const = 0;   

    // The derivative of link_inv.
    virtual double dlink_inv(double eta) const = 0;

    // Second derivative of link_inv.
    virtual double ddlink_inv(double eta) const = 0;

    int nlevels() const {
      return cutpoint_vector().size() + 2;
    }
    
    //---------------------------------------------------------------------------
    // Access to model parameters.
    //---------------------------------------------------------------------------
    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;

    // The vector of cutpoints.  Vector elements must 
    Ptr<VectorParams> Cutpoints_prm();
    const Ptr<VectorParams> Cutpoints_prm() const;

    // Access to regression coefficients is inherited from GlmModel.

    // The minimal vector of model cutpoints.  Cutpoints are in increasing
    // order, and there are nlevels() - 2 of them.
    const Vector &cutpoint_vector() const {
      return Cutpoints_prm()->value();
    }
    void set_cutpoint_vector(const Vector &minimal_cutpoints);
    void set_cutpoint(int index, double value);
    
    // An observed 'y' implies a latent 'z' between lower_cutpoint(y)
    // and upper_cutpoint(y).
    double upper_cutpoint(int y) const;
    double lower_cutpoint(int y) const;

    //---------------------------------------------------------------------------
    // Computing log likelihood.
    //---------------------------------------------------------------------------
    
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
    OrdinalCutpointLogLikelihood cutpoint_log_likelihood() const;

    void initialize_params() override;
    void initialize_params(const Vector &counts);

    Vector CDF(const Vector &x) const;  // Pr(Y<y)

    double pdf(const Data*, bool logscale) const override;
    double pdf(const Ptr<OrdinalRegressionData> &, bool) const;
    double pdf(const OrdinalData &y, const Vector &x, bool logscale) const;
    double pdf(uint y, const Vector &x, bool logscale) const;

    Ptr<OrdinalRegressionData> sim(RNG &rng = GlobalRng::rng);

    // Check whether a candidate vector of cutpoints satisfies the necessary
    // constraints.  All elements of Delta must be positive, and Delta[i] <
    // Delta[i+1] for all i.
    //
    // Returns:
    //   true if the constraints are satisfied.
    bool check_cutpoints(const Vector &cutpoints) const;

    // interface is complicated
    double full_loglike(
        const Vector &beta,
        const Vector &cutpoints,
        Vector &beta_gradient,
        Vector &cutpoint_gradient,
        Matrix &beta_Hessian,
        Matrix &cutpoint_Hessian,
        Matrix &cross_Hessian,
        uint nderiv,
        bool use_beta_derivs,
        bool use_cutpoint_derivs) const;
    
   private:
    // Simulate latent variable from the link distribution.
    virtual double simulate_latent_variable(
        RNG &rng = GlobalRng::rng) const = 0;

    //---------------------------------------------------------------------------
    // Data section.
    //---------------------------------------------------------------------------
    Ptr<CatKeyBase> simulation_key_;

  };

  //===========================================================================
  class OrdinalLogitModel : public OrdinalCutpointModel {
   public:
    OrdinalLogitModel(int xdim, int nlevels)
        : OrdinalCutpointModel(xdim, nlevels) {}
    
    OrdinalLogitModel(const Vector &beta, const Vector &cutpoints)
        : OrdinalCutpointModel(beta, cutpoints) {}

    OrdinalLogitModel(const Vector &beta, Selector &inclusion, const Vector &cutpoints)
        : OrdinalCutpointModel(beta, inclusion, cutpoints) {}

    OrdinalLogitModel(const Selector &inclusion, uint Maxscore)
        : OrdinalCutpointModel(inclusion, Maxscore) {}

    OrdinalLogitModel(const Matrix &X, const Vector &y)
        : OrdinalCutpointModel(X, y) {}

    OrdinalLogitModel * clone() const override {
      return new OrdinalLogitModel(*this);
    }

    // Link function, inverse, and derivative.
    double link(double prob) const override {return qlogis(prob);}
    double link_inv(double eta) const override { return plogis(eta); }
    double dlink_inv(double eta) const override { return dlogis(eta); }
    double ddlink_inv(double eta) const override;

   private:
    // Simulate the unconstrained zero-mean error term.
    double simulate_latent_variable(RNG &rng = GlobalRng::rng) const override {
      return rlogis_mt(rng);
    }
  };

  //===========================================================================
  class OrdinalProbitModel : public OrdinalCutpointModel {
   public:
    OrdinalProbitModel(int xdim, int nlevels)
        : OrdinalCutpointModel(xdim, nlevels) {}
    
    OrdinalProbitModel(const Vector &beta, const Vector &cutpoints)
        : OrdinalCutpointModel(beta, cutpoints) {}

    OrdinalProbitModel(const Vector &beta, Selector &inclusion, const Vector &cutpoints)
        : OrdinalCutpointModel(beta, inclusion, cutpoints) {}

    OrdinalProbitModel(const Selector &inclusion, uint Maxscore)
        : OrdinalCutpointModel(inclusion, Maxscore) {}

    OrdinalProbitModel(const Matrix &X, const Vector &y)
        : OrdinalCutpointModel(X, y) {}

    OrdinalProbitModel * clone() const override {
      return new OrdinalProbitModel(*this);
    }

    // Link function, inverse, and derivative.
    double link(double prob) const override {return qnorm(prob);}
    double link_inv(double eta) const override { return pnorm(eta); }
    double dlink_inv(double eta) const override { return dnorm(eta); }
    double ddlink_inv(double eta) const override;

   private:
    // Simulate the unconstrained zero-mean error term.
    double simulate_latent_variable(RNG &rng = GlobalRng::rng) const override {
      return rnorm_mt(rng);
    }
  };

  
}  // namespace BOOM

#endif  // ORDINAL_CUTPOINT_MODEL_HPP

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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#ifndef POISSON_REGRESSION_MODEL_HPP
#define POISSON_REGRESSION_MODEL_HPP

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/PoissonRegressionData.hpp"
#include "Models/ModelTypes.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"

namespace BOOM {

  // A PoissonRegressionModel describes a non-negative integer
  // response y ~ Poisson(E exp(beta*x)), where E is an exposure.
  class PoissonRegressionModel : public GlmModel,
                                 public NumOptModel,
                                 virtual public MixtureComponent,
                                 public ParamPolicy_1<GlmCoefs>,
                                 public IID_DataPolicy<PoissonRegressionData>,
                                 public PriorPolicy {
   public:
    explicit PoissonRegressionModel(int xdim);
    explicit PoissonRegressionModel(const Vector &beta);

    // Use this constructor if the model is supposed to share its
    // coefficient parameter with another model.
    explicit PoissonRegressionModel(const Ptr<GlmCoefs> &beta);

    PoissonRegressionModel *clone() const override;

    GlmCoefs &coef() override;
    const GlmCoefs &coef() const override;
    Ptr<GlmCoefs> coef_prm() override;
    const Ptr<GlmCoefs> coef_prm() const override;

    // The dimension of the arguments to Loglike and log_likelihood is
    // the number of included coefficients.
    double Loglike(const Vector &beta, Vector &g, Matrix &h,
                   uint nd) const override;

    double log_likelihood() const override {
      Vector g;
      Matrix h;
      return Loglike(Beta(), g, h, 0);
    }

    // Log likelihood function.
    // Args:
    //   beta: The vector of included coefficients (i.e. the dimension
    //     matches that of included_coefficients()).
    //   gradient: If non-NULL the gradient is computed and output
    //     here.  If NULL then no derivative computations are made.
    //   hessian: If hessian and gradient are both non-NULL the
    //     hessian is computed and output here.  If NULL then the
    //     hessian is not computed.
    //   reset_derivatives: If true then a non-NULL gradient or
    //     hessian will be resized and set to zero.  If false then a
    //     non-NULL gradient or hessian will have derivatives of
    //     log-liklihood added to its input value.  It is an error if
    //     reset_derivatives is false and the wrong-sized non-NULL
    //     argument is passed.
    //
    // Returns:
    //   The value of log likelihood at the supplied beta.
    double log_likelihood(const Vector &beta, Vector *gradient = nullptr,
                          Matrix *hessian = nullptr,
                          bool reset_derivatives = true) const;
    // mle() optimizes over the set of included coefficients.
    void mle() override;

    double pdf(const Data *, bool logscale) const override;
    double logp(const PoissonRegressionData &data) const;
    int number_of_observations() const override { return dat().size(); }
  };

}  // namespace BOOM

#endif  // POISSON_REGRESSION_MODEL_HPP

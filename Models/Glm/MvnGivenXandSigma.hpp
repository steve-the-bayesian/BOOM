// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_MVN_GIVEN_X_AND_SIGMA_HPP
#define BOOM_MVN_GIVEN_X_AND_SIGMA_HPP

#include "Models/MvnBase.hpp"
#include "Models/ParamTypes.hpp"
#include "Models/Policies/IID_DataPolicy.hpp"
#include "Models/Policies/ParamPolicy_2.hpp"
#include "Models/Policies/PriorPolicy.hpp"

#include "Models/Glm/Glm.hpp"
#include "Models/Glm/RegressionModel.hpp"

#include <functional>

namespace BOOM {

  // A conjugate prior distribution for the coefficients of a linear
  // regression model.  The conditional distribution of the regression
  // coefficients beta is
  //
  //     beta | X,sigsq ~ N(b, sigsq * (Lambda^{-1} + (kappa * V / n)))
  //
  // where V^{-1} = (1-w) XTX + w Diag(XTX).  The parameters of this model are
  // the vector b (the prior mean of the regression coefficients) and the scalar
  // kappa which can be thought of as a prior sample size.  The matrix Lambda is
  // a diagonal matrix of non-negative numbers, XTX is the cross product matrix
  // from the regression model, and Diag(XTX) is the diagonal matrix with
  // elements from the diagonal of XTX.
  //
  // The prior precision of beta is V^{-1} * kappa / (n * sigma^2).  The main
  // component of V^{-1} is XTX, which is the precision (or information) from
  // the data in a regression model, so dividing by n gives the average
  // information for a single observation.  If Lambda = 0 and w = 0 then kappa
  // could be interpreted as the number of observations worth of data to be
  // assigned as a weight to the prior guess b.  Moving w away from zero
  // averages XTX with its diagonal.  This helps keep the prior proper in cases
  // where X is less than full rank.  Likewise, adding a positive digonal
  // element Lambda keeps the prior propoer in case some columns of X have zero
  // variance, or in the case where X contains no data.
  class MvnGivenXandSigma : public MvnBase,
                            public ParamPolicy_2<VectorParams, UnivParams>,
                            public IID_DataPolicy<GlmCoefs>,
                            public PriorPolicy {
   public:
    // In this constructor, Lambda is taken to be zero.
    // Args:
    //   model: The regression model whose coefficients are to be modeled.  The
    //     model is also the source of the sigsq residual variance parameter,
    //     and the XTX matrix.
    //   prior_mean:  The prior mean (denoted 'b' above).
    //   prior_sample_size:  The prior sample size (denoted 'kappa' above).
    //   additional_prior_precision: The constant to add to the diagonal of the
    //     prior precision matrix.  Denoted 'Lambda' above.
    //   diagonal_weight: The weight to use on the diagonal of XTX when it is
    //     averaged with XTX (which gets 1 - diagonal_weight).
    MvnGivenXandSigma(RegressionModel *model,
                      const Ptr<VectorParams> &prior_mean,
                      const Ptr<UnivParams> &prior_sample_size,
                      const Vector &Lambda = Vector(0),
                      double diagonal_weight = 0);

    // Use this constructor if you want to specify XTX and sigsq
    // without passing a regression model.
    // Args:
    //   residual variance parameter, and the XTX matrix.
    //   prior_mean:  The prior mean (denoted 'b' above).
    //   prior_sample_size:  The prior sample size (denoted 'kappa' above).
    //   sigsq:  The residual variance in the regression model.
    //   XTX:  The cross product matrix from the regression model.
    //   sample_size: The number of observations contained in X.  This is
    //     denoted 'n' above.
    //   additional_prior_precision: The constant to add to the diagonal of the
    //     prior precision matrix.  Denoted 'Lambda' above.
    //   diagonal_weight: The weight to use on the diagonal of XTX
    MvnGivenXandSigma(const Ptr<VectorParams> &prior_mean,
                      const Ptr<UnivParams> &prior_sample_size,
                      const Ptr<UnivParams> &sigsq, const SpdMatrix &XTX,
                      double sample_size, const Vector &Lambda = Vector(0),
                      double diagonal_weight = 0);

    MvnGivenXandSigma(const MvnGivenXandSigma &rhs);
    MvnGivenXandSigma *clone() const override;

    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override;

    // The value of the prior sample size parameter (denoted kappa
    // above).
    double prior_sample_size() const;

    // The number of observations in the underlying regression model.
    double data_sample_size() const;

    const Ptr<VectorParams> Mu_prm() const;
    const Ptr<UnivParams> Kappa_prm() const;
    Ptr<VectorParams> Mu_prm();
    Ptr<UnivParams> Kappa_prm();
    double diagonal_weight() const;

    Vector sim(RNG &rng = GlobalRng::rng) const override;

    // An observer to be called whenever the underlying regression
    // model changes its residual variance parameter, or its data.
    void observe_changes() { current_ = false; }

   private:
    Ptr<UnivParams> sigsq_;
    std::function<const SpdMatrix(void)> compute_xtx_;
    std::function<double(void)> data_sample_size_;
    Vector additional_prior_precision_;
    double diagonal_weight_;

    mutable Ptr<SpdParams> ivar_;
    mutable bool current_;
    void set_ivar() const;  // logical constness
  };
}  // namespace BOOM
#endif  // BOOM_MVN_GIVEN_X_AND_SIGMA_HPP

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2010 Steven L. Scott

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

#ifndef BOOM_MVN_GIVEN_SCALAR_SIGMA_HPP_
#define BOOM_MVN_GIVEN_SCALAR_SIGMA_HPP_

#include "Models/MvnBase.hpp"
#include "Models/Policies/ParamPolicy_1.hpp"
#include "Models/Policies/PriorPolicy.hpp"
#include "Models/Policies/SufstatDataPolicy.hpp"
#include "Models/SpdData.hpp"

namespace BOOM {

  // This model is intended for use as a conditional prior
  // distribution for regression coefficients in "least squares"
  // regression problems.  The model is
  //
  // beta | sigsq ~ N(b, sigsq * Omega)
  //
  // Omega inverse will typically be some multiple of the "XTX" cross
  // product matrix in the regression model, so the constructors take
  // the inverse of Omega as an argument.  Omega is a fixed constant in
  // this model, which might make it a poor fit for hierarchical models
  // where the degree of shrinkage is to be learned across groups.
  class MvnGivenScalarSigmaBase : public MvnBase {
   public:
    explicit MvnGivenScalarSigmaBase(const Ptr<UnivParams> &sigsq);
    double sigsq() const;

    virtual const SpdMatrix &unscaled_precision() const = 0;

   private:
    // sigsq_ is a pointer to the residual variance parameter, e.g. in
    // a regression model.
    Ptr<UnivParams> sigsq_;
  };

  //======================================================================
  // The concrete class to use with arbitrary "Omega" values.
  class MvnGivenScalarSigma : public MvnGivenScalarSigmaBase,
                              public LoglikeModel,
                              public ParamPolicy_1<VectorParams>,
                              public SufstatDataPolicy<VectorData, MvnSuf>,
                              public PriorPolicy {
   public:
    MvnGivenScalarSigma(const SpdMatrix &ominv, const Ptr<UnivParams> &sigsq);
    MvnGivenScalarSigma(const Vector &mean, const SpdMatrix &ominv,
                        const Ptr<UnivParams> &sigsq);

    MvnGivenScalarSigma(const MvnGivenScalarSigma &rhs);
    MvnGivenScalarSigma *clone() const override;

    Ptr<VectorParams> Mu_prm();
    const Ptr<VectorParams> Mu_prm() const;

    uint dim() const override;
    const Vector &mu() const override;

    // Sigma refers to the actual variance matrix of beta given sigma
    // and Omega, i.e. Omega * sigsq.  siginv and ldsi refer to its
    // inverse and the log of the determinant of its inverse.
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    double ldsi() const override;

    // Omega refers to the proportional variance matrix of beta
    // (i.e. not multiplied by sigsq).  ominv and ldoi refer to the
    // inverse of this matrix and the log of the determinant of the
    // inverse.
    const SpdMatrix &Omega() const;
    const SpdMatrix &ominv() const;
    const SpdMatrix &unscaled_precision() const override {return ominv();}
    double ldoi() const;

    void set_mu(const Vector &);

    void set_unscaled_precision(const SpdMatrix &omega_inverse);

    void mle() override;
    double loglike(const Vector &mu_ominv) const override;
    double pdf(const Ptr<Data> &dp, bool) const;

   private:
    // ominv_ is stored as SpdParams instead of as a raw SpdMatrix because
    // SpdParams keeps track of the matrix, its inverse, and its log
    // determinant.
    SpdData omega_;

    // The following is workspace used to comply with the
    // return-by-reference interface promised by MvnBase for Sigma(),
    // siginv(), and ldsi().
    mutable SpdMatrix wsp_;
  };

}  // namespace BOOM
#endif  // BOOM_MVN_GIVEN_SCALAR_SIGMA_HPP_

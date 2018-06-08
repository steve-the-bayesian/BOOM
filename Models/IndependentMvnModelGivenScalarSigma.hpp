// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_INDEPENDENT_MVN_MODEL_GIVEN_SCALAR_SIGMA_HPP_
#define BOOM_INDEPENDENT_MVN_MODEL_GIVEN_SCALAR_SIGMA_HPP_

#include "Models/IndependentMvnModel.hpp"
#include "Models/MvnGivenScalarSigma.hpp"

namespace BOOM {

  // This class is intended to serve as a prior distribution for a set
  // of regression coefficients in a linear regression model.  The
  // model states beta ~ N(b, sigsq * V), where sigsq is the residual
  // variance from the linear regression model, and V is a diagonal
  // matrix (so that the elements of beta are independent).
  //
  // The class stores 'sigsq' using a Ptr, which must be obtained from
  // the linear regression model.
  class IndependentMvnModelGivenScalarSigma
      : public MvnGivenScalarSigmaBase,
        public ParamPolicy_2<VectorParams, VectorParams>,
        public IID_DataPolicy<VectorData>,
        public PriorPolicy {
   public:
    // Args:
    //   prior_mean:  The mean of the distribution.
    //   unscaled_variance_diagonal: A vector positive numbers
    //     representing the diagonal of the unscaled variance matrix.
    //     Multiplying each element by sigsq gives the element-wise
    //     variance of the random variable being modeled.
    //   sigsq: The residual variance parameter.  This is not
    //     considered to be a "parameter" of this model, in the sense
    //     that it is not managed by a ParamPolicy.
    IndependentMvnModelGivenScalarSigma(const Vector &prior_mean,
                                        const Vector &unscaled_variance_diagonal,
                                        const Ptr<UnivParams> &sigsq);

    // Args:
    //   prior_mean:  The mean of the distribution.
    //   unscaled_variance_diagonal: A vector positive numbers
    //     representing the diagonal of the unscaled variance matrix.
    //     Multiplying each element by sigsq gives the element-wise
    //     variance of the random variable being modeled.
    //   sigsq: The residual variance parameter.  This is not
    //     considered to be a "parameter" of this model, in the sense
    //     that it is not managed by a ParamPolicy.
    IndependentMvnModelGivenScalarSigma(
        const Ptr<VectorParams> &prior_mean,
        const Ptr<VectorParams> &unscaled_variance_diagonal,
        const Ptr<UnivParams> &sigsq);

    IndependentMvnModelGivenScalarSigma *clone() const override;

    double Logp(const Vector &x, Vector &gradient, Matrix &hessian,
                uint nderiv) const override;
    const Vector &mu() const override;
    const SpdMatrix &Sigma() const override;
    const SpdMatrix &siginv() const override;
    const SpdMatrix &unscaled_precision() const override;
    
    double ldsi() const override;
    Vector sim(RNG &rng = GlobalRng::rng) const override;

    // unscaled_variance_diagonal() * sigsq() is the diagonal of
    // Sigma().
    const Vector &unscaled_variance_diagonal() const;
    double sd_for_element(int i) const;

   private:
    mutable SpdMatrix sigma_scratch_;
  };

}  // namespace BOOM

#endif  //  BOOM_INDEPENDENT_MVN_MODEL_GIVEN_SCALAR_SIGMA_HPP_

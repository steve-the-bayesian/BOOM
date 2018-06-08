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

#include "Models/IndependentMvnModelGivenScalarSigma.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using IMMGS = BOOM::IndependentMvnModelGivenScalarSigma;
  }

  IMMGS::IndependentMvnModelGivenScalarSigma(
      const Vector &prior_mean,
      const Vector &unscaled_variance_diagonal,
      const Ptr<UnivParams> &sigsq)
      : MvnGivenScalarSigmaBase(sigsq),
        ParamPolicy(new VectorParams(prior_mean),
                    new VectorParams(unscaled_variance_diagonal)) {}

  IMMGS::IndependentMvnModelGivenScalarSigma(
      const Ptr<VectorParams> &prior_mean,
      const Ptr<VectorParams> &unscaled_variance_diagonal,
      const Ptr<UnivParams> &sigsq)
      : MvnGivenScalarSigmaBase(sigsq),
        ParamPolicy(prior_mean, unscaled_variance_diagonal) {}

  IndependentMvnModelGivenScalarSigma *IMMGS::clone() const {
    return new IndependentMvnModelGivenScalarSigma(*this);
  }

  double IMMGS::Logp(const Vector &x, Vector &gradient, Matrix &hessian,
                     uint nderiv) const {
    double ans = 0;
    if (nderiv > 0) {
      gradient = 0;
      if (nderiv > 1) {
        hessian = 0;
      }
    }
    const Vector &mu(this->mu());
    Vector v = unscaled_variance_diagonal() * sigsq();
    for (int i = 0; i < x.size(); ++i) {
      ans += dnorm(x[i], mu[i], sqrt(v[i]), true);
      if (nderiv > 0) {
        gradient[i] -= -(x[i] - mu[i]) / v[i];
        if (nderiv > 1) {
          hessian(i, i) -= 1.0 / v[i];
        }
      }
    }
    return ans;
  }

  const Vector &IMMGS::mu() const { return prm1_ref().value(); }

  const SpdMatrix &IMMGS::Sigma() const {
    sigma_scratch_.resize(dim());
    sigma_scratch_.diag() = unscaled_variance_diagonal();
    sigma_scratch_.diag() *= sigsq();
    return sigma_scratch_;
  }

  const SpdMatrix &IMMGS::siginv() const {
    sigma_scratch_.resize(dim());
    sigma_scratch_.diag() = 1.0 / unscaled_variance_diagonal();
    sigma_scratch_.diag() /= sigsq();
    return sigma_scratch_;
  }

  const SpdMatrix &IMMGS::unscaled_precision() const {
    sigma_scratch_.resize(dim());
    sigma_scratch_.diag() = 1.0 / unscaled_variance_diagonal();
    return sigma_scratch_;
  }
  
  double IMMGS::ldsi() const {
    double ans = -dim() * log(sigsq());
    const Vector &v(unscaled_variance_diagonal());
    for (int i = 0; i < dim(); ++i) {
      ans -= log(v[i]);
    }
    return ans;
  }

  Vector IMMGS::sim(RNG &rng) const {
    Vector ans(dim());
    double sigma = sqrt(sigsq());
    const Vector &v(unscaled_variance_diagonal());
    const Vector &mu(this->mu());
    for (int i = 0; i < dim(); ++i) {
      ans[i] = rnorm_mt(rng, mu[i], sigma * sqrt(v[i]));
    }
    return ans;
  }

  const Vector &IMMGS::unscaled_variance_diagonal() const {
    return prm2_ref().value();
  }

  double IMMGS::sd_for_element(int i) const {
    return sqrt(sigsq() * unscaled_variance_diagonal()[i]);
  }

}  // namespace BOOM

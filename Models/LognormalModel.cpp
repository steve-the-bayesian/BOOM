// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Models/LognormalModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  LognormalModel::LognormalModel(double mu, double sigma)
      : ParamPolicy(new UnivParams(mu), new UnivParams(sigma * sigma)),
        DataPolicy(new GaussianSuf) {
    if (sigma <= 0) {
      report_error("Standard deviation must be positive.");
    }
  }

  LognormalModel::LognormalModel(const Ptr<UnivParams> &mu,
                                 const Ptr<UnivParams> &sigsq)
      : ParamPolicy(mu, sigsq), DataPolicy(new GaussianSuf) {
    if (sigsq->value() <= 0) {
      report_error("Variance must be positive.");
    }
  }

  LognormalModel *LognormalModel::clone() const {
    return new LognormalModel(*this);
  }

  Ptr<UnivParams> LognormalModel::Mu_prm() { return prm1(); }

  Ptr<UnivParams> LognormalModel::Sigsq_prm() { return prm2(); }

  double LognormalModel::mu() const { return prm1_ref().value(); }

  double LognormalModel::sigsq() const { return prm2_ref().value(); }

  void LognormalModel::set_mu(double mu) { prm1_ref().set(mu); }

  void LognormalModel::set_sigsq(double sigsq) {
    if (sigsq <= 0) {
      report_error("Variance must be positive.");
    }
    prm2_ref().set(sigsq);
  }

  double LognormalModel::mean() const { return exp(mu() + 0.5 * sigsq()); }

  double LognormalModel::variance() const {
    return expm1(sigsq()) * square(mean());
  }

  double LognormalModel::Logp(double x, double &d1, double &d2,
                              uint nderiv) const {
    if (nderiv > 0) {
      double logx = log(x);
      double residual = logx - mu();
      d1 = -1 / x - residual / (sigsq() * x);
      if (nderiv > 1) {
        double xsquare = x * x;
        d2 = (1.0 / xsquare) - (1 - residual) / (sigsq() * xsquare);
      }
    }
    return dlnorm(x, mu(), sigma(), true);
  }

  double LognormalModel::sim(RNG &rng) const {
    return exp(rnorm_mt(rng, mu(), sigma()));
  }

}  // namespace BOOM

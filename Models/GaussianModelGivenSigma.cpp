// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/GaussianModelGivenSigma.hpp"
#include <cmath>
#include "Models/GammaModel.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    using GMGS = BOOM::GaussianModelGivenSigma;
  }  // namespace

  GMGS::GaussianModelGivenSigma(const Ptr<UnivParams> &scaling_variance,
                                double mean, double sample_size)
      : ParamPolicy(new UnivParams(mean), new UnivParams(sample_size)),
        scaling_variance_(scaling_variance) {}

  GMGS *GMGS::clone() const { return new GMGS(*this); }

  Ptr<UnivParams> GMGS::Mu_prm() { return prm1(); }
  Ptr<UnivParams> GMGS::Kappa_prm() { return prm2(); }
  const Ptr<UnivParams> GMGS::Mu_prm() const { return prm1(); }
  const Ptr<UnivParams> GMGS::Kappa_prm() const { return prm2(); }

  void GMGS::set_params(double mu0, double kappa) {
    set_mu(mu0);
    set_kappa(kappa);
  }

  void GMGS::set_scaling_variance(const Ptr<UnivParams> &scaling_variance) {
    scaling_variance_ = scaling_variance;
  }

  double GMGS::mu() const { return prm1_ref().value(); }
  void GMGS::set_mu(double mu0) { Mu_prm()->set(mu0); }

  double GMGS::kappa() const { return prm2_ref().value(); }
  void GMGS::set_kappa(double kappa) { Kappa_prm()->set(kappa); }

  double GMGS::scaling_variance() const {
    if (!scaling_variance_) {
      report_error("Scaling variance is not set.");
    }
    return scaling_variance_->value();
  }

  double GMGS::sigsq() const { return scaling_variance() / kappa(); }

  double GMGS::Loglike(const Vector &mu_kappa, Vector &g, Matrix &h,
                       uint nderiv) const {
    if (mu_kappa.size() != 2) {
      report_error(
          "Wrong size argument passed to GaussianModelGivenSigma"
          "::Loglike.");
    }
    double sigsq = this->scaling_variance();
    if (sigsq < 0) {
      return negative_infinity();
    }

    double mu = mu_kappa[0];
    double kappa = mu_kappa[1];
    if (kappa <= 0) {
      return negative_infinity();
    }

    const double log2pi = 1.8378770664093453;
    double n = suf()->n();
    double centered_sumsq = suf()->centered_sumsq(mu);
    double ans = .5 * n * (-log2pi + log(kappa) - log(sigsq));
    ans -= .5 * kappa * centered_sumsq / sigsq;

    if (nderiv > 0) {
      double residual_sum = suf()->sum() - n * mu;
      g[0] = kappa * residual_sum / sigsq;
      g[1] = .5 * ((n / kappa) - (centered_sumsq / sigsq));
      if (nderiv > 1) {
        h(0, 0) = -n * kappa / sigsq;
        h(1, 0) = h(0, 1) = residual_sum / sigsq;
        h(1, 1) = -0.5 * n / square(kappa);
      }
    }
    return ans;
  }

  void GMGS::mle() {
    double n = suf()->n();
    double sample_mean = n < 1 ? 0 : ybar();
    double sigma_hat_squared = sample_var() * (n - 1) / n;
    double kappa = (n <= 1) ? 1.0 : scaling_variance() / sigma_hat_squared;
    set_params(sample_mean, kappa);
  }

}  // namespace BOOM

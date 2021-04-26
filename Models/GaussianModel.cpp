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
#include "Models/GaussianModel.hpp"
#include "Models/GammaModel.hpp"
#include "Models/GaussianModelGivenSigma.hpp"
#include "Models/PosteriorSamplers/PosteriorSampler.hpp"

#include <cmath>
#include <typeinfo>
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {
  using GM = BOOM::GaussianModel;

  GaussianModel::GaussianModel(double mean, double sd)
      : ParamPolicy(new UnivParams(mean), new UnivParams(sd * sd)) {}

  GaussianModel::GaussianModel(const std::vector<double> &v)
      : GaussianModelBase(v),
        ParamPolicy(new UnivParams(0), new UnivParams(1)) {
    mle();
  }

  GaussianModel::GaussianModel(const GaussianModel &rhs)
      : Model(rhs),
        GaussianModelBase(rhs),
        ParamPolicy(rhs),
        PriorPolicy(rhs) {}

  GM *GM::clone() const { return new GM(*this); }

  Ptr<UnivParams> GM::Mu_prm() { return prm1(); }
  Ptr<UnivParams> GM::Sigsq_prm() { return prm2(); }
  const Ptr<UnivParams> GM::Mu_prm() const { return prm1(); }
  const Ptr<UnivParams> GM::Sigsq_prm() const { return prm2(); }

  void GM::set_params(double mu, double sigsq) {
    set_mu(mu);
    set_sigsq(sigsq);
  }
  void GM::set_mu(double m) { Mu_prm()->set(m); }
  void GM::set_sigsq(double s) { Sigsq_prm()->set(s); }

  double GM::mu() const { return Mu_prm()->value(); }
  double GM::sigsq() const { return Sigsq_prm()->value(); }
  double GM::sigma() const { return sqrt(sigsq()); }

  void GaussianModel::mle() {
    double n = suf()->n();
    if (n == 0) {
      set_params(0, 1);
      return;
    }

    double m = ybar();
    if (n == 1) {
      set_params(ybar(), 1.0);
      return;
    }
    double v = sample_var() * (n - 1) / n;
    set_params(m, v);
  }

  double GaussianModel::Loglike(const Vector &mu_sigsq, Vector &g, Matrix &h,
                                uint nd) const {
    double sigsq = mu_sigsq[1];
    if (sigsq < std::numeric_limits<double>::min()) {
      return BOOM::negative_infinity();
    }

    double mu = mu_sigsq[0];
    const double log2pi = 1.8378770664093453;
    double n = suf()->n();
    double sumsq = suf()->sumsq();
    double sum = suf()->sum();
    double SS = (sumsq + (-2 * sum + n * mu) * mu);
    double ans = -0.5 * (n * (log2pi + log(sigsq)) + SS / sigsq);

    if (nd > 0) {
      double sig4 = sigsq * sigsq;
      g[0] = (sum - n * mu) / sigsq;
      g[1] = -0.5 * n / sigsq + 0.5 * SS / sig4;
      if (nd > 1) {
        h(0, 0) = -n / sigsq;
        h(1, 0) = h(0, 1) = -(sum - n * mu) / sig4;
        h(1, 1) = (n / 2 - SS / sigsq) / sig4;
      }
    }
    return ans;
  }

}  // namespace BOOM

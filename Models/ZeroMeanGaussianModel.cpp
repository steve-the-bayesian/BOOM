// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008-2011 Steven L. Scott

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

#include "Models/ZeroMeanGaussianModel.hpp"
#include "Models/GammaModel.hpp"
#include "cpputil/math_utils.hpp"

namespace BOOM {

  typedef ZeroMeanGaussianModel ZGM;

  ZGM::ZeroMeanGaussianModel(double sigma)
      : ParamPolicy(new UnivParams(sigma * sigma)) {}

  ZGM::ZeroMeanGaussianModel(const std::vector<double> &y)
      : GaussianModelBase(y), ParamPolicy(new UnivParams(1.0)) {
    mle();
  }

  ZGM *ZGM::clone() const { return new ZGM(*this); }

  void ZGM::set_sigsq(double s2) { Sigsq_prm()->set(s2); }

  Ptr<UnivParams> ZGM::Sigsq_prm() { return ParamPolicy::prm(); }

  const Ptr<UnivParams> ZGM::Sigsq_prm() const { return ParamPolicy::prm(); }

  double ZGM::sigsq() const { return Sigsq_prm()->value(); }
  double ZGM::sigma() const { return sqrt(sigsq()); }

  void ZGM::mle() {
    double n = suf()->n();
    double ss = suf()->sumsq();
    if (n > 0)
      set_sigsq(ss / n);
    else
      set_sigsq(1.0);
  }

  double ZGM::log_likelihood(double sigsq, double *d1, double *d2) const {
    if (sigsq < 0) return BOOM::negative_infinity();
    const double log2pi = 1.8378770664093453;
    double n = suf()->n();
    double sumsq = suf()->sumsq();
    double ans = -0.5 * (n * (log2pi + log(sigsq)) + sumsq / sigsq);
    if (d1) {
      double sig4 = sigsq * sigsq;
      *d1 = .5 * ((sumsq / sig4) - (n / sigsq));
      if (d2) {
        *d2 = (n / 2 - sumsq / sigsq) / sig4;
      }
    }
    return ans;
  }

  double ZGM::Loglike(const Vector &sigsq_vec, Vector &g, Matrix &h,
                      uint nd) const {
    return log_likelihood(sigsq_vec[0], nd > 0 ? &g[0] : nullptr,
                          nd > 1 ? &h(0, 0) : nullptr);
  }
}  // namespace BOOM

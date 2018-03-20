// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "Models/ZeroMeanMvnModel.hpp"
#include "Models/PosteriorSamplers/ZeroMeanMvnConjSampler.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef ZeroMeanMvnModel ZMMM;

  ZMMM::ZeroMeanMvnModel(int dim)
      : ParamPolicy(new SpdParams(dim)),
        DataPolicy(new MvnSuf(dim)),
        mu_(dim, 0.0) {}

  ZMMM *ZMMM::clone() const { return new ZMMM(*this); }
  const Vector &ZMMM::mu() const { return mu_; }
  const SpdMatrix &ZMMM::Sigma() const { return prm()->var(); }
  const SpdMatrix &ZMMM::siginv() const { return prm()->ivar(); }
  double ZMMM::ldsi() const { return prm()->ldsi(); }
  void ZMMM::set_Sigma(const SpdMatrix &v) { prm()->set_var(v); }
  void ZMMM::set_siginv(const SpdMatrix &ivar) { prm()->set_ivar(ivar); }
  Ptr<SpdParams> ZMMM::Sigma_prm() { return prm(); }
  const Ptr<SpdParams> ZMMM::Sigma_prm() const { return prm(); }

  void ZMMM::mle() {
    double n = suf()->n();
    if (n < 1) {
      report_error(
          "Too few degrees of freedom to compute ML in "
          "ZeroMeanGaussianModel::mle()");
    }
    set_Sigma(suf()->center_sumsq(mu_) / (n - 1));
  }

  double ZMMM::pdf(const Ptr<Data> &dp, bool logscale) const {
    Ptr<VectorData> dpp = DAT(dp);
    return dmvn_zero_mean(dpp->value(), siginv(), ldsi(), logscale);
  }

  double ZMMM::loglike(const Vector &siginv_triangle) const {
    const double log2pi = 1.83787706641;
    double dim = mu_.size();
    double n = suf()->n();
    const Vector ybar = suf()->ybar();
    const SpdMatrix sumsq = suf()->center_sumsq();

    SpdMatrix siginv(dim);
    siginv.unvectorize(siginv_triangle, true);

    double qform = n * (siginv.Mdist(ybar));
    qform += traceAB(siginv, sumsq);

    double nc = 0.5 * n * (-dim * log2pi + siginv.logdet());

    double ans = nc - .5 * qform;
    return ans;
  }
}  // namespace BOOM

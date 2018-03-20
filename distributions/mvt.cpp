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

#include <cmath>
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "distributions.hpp"

#include "cpputil/Constants.hpp"

namespace BOOM {
  //======================================================================
  double dmvt(const Vector &x, const Vector &mu, const SpdMatrix &Siginv,
              double nu, bool logscale) {
    double ldsi = Siginv.logdet();
    return dmvt(x, mu, Siginv, nu, ldsi, logscale);
  }
  //======================================================================
  double dmvt(const Vector &x, const Vector &mu, const SpdMatrix &Siginv,
              double nu, double ldsi, bool logscale) {
    long dim = mu.size();
    double nc = lgamma((nu + dim) / 2.0) + .5 * ldsi - lgamma(nu / 2.0) -
                (.5 * dim) * (log(nu) + Constants::log_pi);
    double delta = Siginv.Mdist(x, mu);
    double ans = nc - .5 * (nu + dim) * (::log1p(delta / nu));
    return logscale ? ans : exp(ans);
  }
  //======================================================================

  Vector rmvt(const Vector &mu, const SpdMatrix &Sigma, double nu) {
    return rmvt_mt(GlobalRng::rng, mu, Sigma, nu);
  }

  Vector rmvt_mt(RNG &rng, const Vector &mu, const SpdMatrix &Sigma,
                 double nu) {
    double w = rgamma_mt(rng, nu / 2, nu / 2);
    return rmvn_mt(rng, mu, Sigma / w);
  }

  Vector rmvt_ivar(const Vector &mu, const SpdMatrix &ivar, double nu) {
    return rmvt_ivar_mt(GlobalRng::rng, mu, ivar, nu);
  }

  Vector rmvt_ivar_mt(RNG &rng, const Vector &mu, const SpdMatrix &ivar,
                      double nu) {
    double w = rgamma_mt(rng, nu / 2, nu / 2);
    return rmvn_ivar_mt(rng, mu, w * ivar);
  }
}  // namespace BOOM

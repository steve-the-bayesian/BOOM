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
#include <cmath>

#include "distributions.hpp"

#include <algorithm>
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include <iostream>

namespace BOOM {

  double dmatrix_normal_ivar(const Matrix &Y,
                             const Matrix &Mu,
                             const SpdMatrix &Ominv,
                             const SpdMatrix &Siginv,
                             bool logscale) {
    double ldoi = Ominv.logdet();
    double ldsi = Siginv.logdet();
    return dmatrix_normal_ivar(Y, Mu, Siginv, ldsi, Ominv, ldoi, logscale);
  }

  double dmatrix_normal_ivar(const Matrix &Y,
                             const Matrix &Mu,
                             const SpdMatrix &Ominv,
                             double ldoi,
                             const SpdMatrix &Siginv,
                             double ldsi,
                             bool logscale) {
    Matrix E = Y - Mu;
    double qform = traceAtB(Ominv * E, E * Siginv);

    // qform = vec(Y-Mu)^T (Siginv \otimes Ominv) vec(Y-Mu)
    //  = tr(E^T Ominv E Siginv)    see Harville (1997) 16.2.15

    uint xdim = Y.nrow();
    uint ydim = Y.ncol();
    double logdet_ivar = ydim * ldoi + xdim * ldsi;
    // xsize * ydeterm. + oppositex

    const double log2pi = 1.83787706641;

    uint n = xdim * ydim;
    double ans = -.5 * n * log2pi + .5 * logdet_ivar - .5 * qform;
    return logscale ? ans : exp(ans);
  }

  Matrix rmatrix_normal_ivar(const Matrix &Mu, const SpdMatrix &Ominv,
                             const SpdMatrix &Siginv) {
    return rmatrix_normal_ivar_mt(GlobalRng::rng, Mu, Ominv, Siginv);
  }

  Matrix rmatrix_normal_ivar_mt(RNG &rng, const Matrix &Mu,
                                const SpdMatrix &Ominv,
                                const SpdMatrix &Siginv) {
    uint xdim = Mu.nrow();
    uint ydim = Mu.ncol();
    Matrix Z(xdim, ydim);
    double *zdata = Z.data();
    for (uint i = 0; i < xdim * ydim; ++i) {
      zdata[i] = rnorm_mt(rng);
    }

    Matrix Ominv_U(t(Cholesky(Ominv).getL()));
    Matrix Lsig(Linv(Cholesky(Siginv).getL()));

    Matrix ans = Mu + Usolve(Ominv_U, Z) * Lsig;
    return ans;
  }

}  // namespace BOOM

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

#include <algorithm>
#include <cmath>
#include "LinAlg/Cholesky.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/DiagonalMatrix.hpp"
#include "distributions.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  Vector rmvn_robust(const Vector &mu, const SpdMatrix &V) {
    return rmvn_robust_mt(GlobalRng::rng, mu, V);
  }
  Vector rmvn_robust_mt(RNG &rng, const Vector &mu, const SpdMatrix &V) {
    uint n = V.nrow();
    Matrix eigenvectors(n, n);
    Vector eigenvalues = eigen(V, eigenvectors);
    for (uint i = 0; i < n; ++i) {
      // We're guaranteed that eigenvalues[i] is real and non-negative.  We
      // can take the absolute value of eigenvalues[i] to guard against
      // spurious negative numbers close to zero.
      eigenvalues[i] = sqrt(fabs(eigenvalues[i])) * rnorm_mt(rng, 0, 1);
    }
    Vector ans(eigenvectors * eigenvalues);
    ans += mu;
    return ans;
  }

  Vector rmvn_L(const Vector &mu, const Matrix &L) {
    return rmvn_L_mt(GlobalRng::rng, mu, L);
  }

  Vector rmvn_L_mt(RNG &rng, const Vector &mu, const Matrix &L) {
    // L is the lower cholesky triangle of Sigma.
    uint n = mu.size();
    Vector wsp(n);
    for (uint i = 0; i < n; ++i) wsp[i] = rnorm_mt(rng, 0, 1);
    return Lmult(L, wsp) + mu;
  }
  //======================================================================
  Vector rmvn(const Vector &mu, const SpdMatrix &V) {
    return rmvn_mt(GlobalRng::rng, mu, V);
  }

  Matrix rmvn_repeated(int sample_size, const SpdMatrix &Sigma) {
    int ydim = Sigma.nrow();
    Matrix ans(sample_size, ydim);
    Matrix L = Sigma.chol();
    for (int i = 0; i < sample_size; ++i) {
      Vector draw(ydim);
      for (int j = 0; j < ydim; ++j) {
        draw[j] = rnorm_mt(GlobalRng::rng, 0, 1);
      }
      ans.row(i) = L * draw;
    }
    return ans;
  }

  Vector rmvn_mt(RNG &rng, const Vector &mu, const SpdMatrix &V) {
    bool okay = true;
    Matrix L = V.chol(okay);
    if (okay) return rmvn_L_mt(rng, mu, L);
    return rmvn_robust_mt(rng, mu, V);
  }
  //======================================================================
  Vector rmvn_mt(RNG &rng, const Vector &mu, const DiagonalMatrix &V) {
    Vector ans(mu);
    const ConstVectorView variances(V.diag());
    for (int i = 0; i < mu.size(); ++i) {
      ans[i] += rnorm_mt(rng, 0, sqrt(variances[i]));
    }
    return ans;
  }
  Vector rmvn(const Vector &mu, const DiagonalMatrix &V) {
    return rmvn_mt(GlobalRng::rng, mu, V);
  }
  //======================================================================
  Vector rmvn_ivar(const Vector &mu, const SpdMatrix &ivar) {
    return rmvn_ivar_mt(GlobalRng::rng, mu, ivar);
  }

  Vector rmvn_ivar_mt(RNG &rng, const Vector &mu, const SpdMatrix &ivar) {
    // Draws a multivariate normal with mean mu and precision matrix
    // ivar.
    bool ok = false;
    Matrix U = ivar.chol(ok).transpose();
    if (!ok) {
      report_error("Cholesky decomposition failed in rmvn_ivar_mt.");
    }
    return rmvn_precision_upper_cholesky_mt(rng, mu, U);
  }

  Vector rmvn_precision_upper_cholesky_mt(
      RNG &rng, const Vector &mu, const Matrix &precision_upper_cholesky) {
    // U is the upper cholesky factor of the inverse variance Matrix
    uint n = mu.size();
    Vector z(n);
    for (uint i = 0; i < n; ++i) z[i] = rnorm_mt(rng, 0, 1);
    //    if precision = L L^T then Sigma = (L^T)^{-1} L^{-1} = U U^T
    return Usolve_inplace(precision_upper_cholesky, z) + mu;
  }

  Vector rmvn_suf(const SpdMatrix &Ivar, const Vector &IvarMu) {
    return rmvn_suf_mt(GlobalRng::rng, Ivar, IvarMu);
  }

  Vector rmvn_suf_mt(RNG &rng, const SpdMatrix &Ivar, const Vector &IvarMu) {
    Cholesky L(Ivar);
    uint n = IvarMu.size();
    Vector z(n);
    for (uint i = 0; i < n; ++i) z[i] = rnorm_mt(rng);
    LTsolve_inplace(L.getL(), z);  // returns LT^-1 z which is ~ N(0, Ivar.inv)
    z += L.solve(IvarMu);
    return z;
  }

  //======================================================================
  Vector &impute_mvn(Vector &observation,
                     const Vector &mean, const SpdMatrix &variance,
                     const Selector &observed, RNG &rng) {
    if (observed.nvars() == observed.nvars_possible()) {
      return observation;
    } else if (observed.nvars() == 0) {
      observation = rmvn_mt(rng, mean, variance);
      return observation;
    }
    if (observation.size() != observed.nvars_possible()) {
      report_error("observation and observed must be the same size.");
    }

    // The distribution we want is N(mu, V), with
    //  V = Sig11 - Sig12 Sig22.inv Sig.21
    // and
    // mu = mu1 - Sig12 Sig22.inv (y2 - mu2)
    // The 1's are missing, and the 2's are observed.
    Selector missing = observed.complement();
    Matrix cross_covariance = missing.select_rows(
        observed.select_cols(variance));
    SpdMatrix observed_precision = observed.select_square(variance).inv();
    Vector mu = missing.select(mean) + cross_covariance * observed_precision
        * (observed.select(observation) - observed.select(mean));
    SpdMatrix V = missing.select_square(variance)
        - sandwich(cross_covariance, observed_precision);
    Vector imputed = rmvn_mt(rng, mu, V);
    observed.fill_missing_elements(observation, imputed);
    return observation;
  }

  //======================================================================
  double dmvn(const Vector &y, const Vector &mu, const SpdMatrix &Siginv,
              double ldsi, bool logscale) {
    const double log2pi = 1.83787706641;
    double n = y.size();
    double ans = 0.5 * (ldsi - Siginv.Mdist(y, mu) - n * log2pi);
    return logscale ? ans : std::exp(ans);
  }

  double dmvn_zero_mean(const Vector &y, const SpdMatrix &Siginv, double ldsi,
                        bool logscale) {
    const double log2pi = 1.83787706641;
    double n = y.size();
    double ans = 0.5 * (ldsi - Siginv.Mdist(y) - n * log2pi);
    return logscale ? ans : std::exp(ans);
  }

  double dmvn(const Vector &y, const Vector &mu, const SpdMatrix &Siginv,
              bool logscale) {
    double ldsi = Siginv.logdet();
    return dmvn(y, mu, Siginv, ldsi, logscale);
  }

}  // namespace BOOM

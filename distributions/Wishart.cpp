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
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"

#include <stdexcept>
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  // returns the Bartlett decomposition of a Wishart matrix of
  // dimension d and nu degrees of freedom
  Matrix WishartTriangle(RNG &rng, int dim, double nu) {
    Matrix ans(dim, dim, 0.0);
    for (int i = 0; i < dim; ++i) {
      ans(i, i) = sqrt(rchisq_mt(rng, nu - i));
      for (int j = 0; j < i; ++j) ans(i, j) = rnorm_mt(rng);
    }
    return ans;
  }

  SpdMatrix rWish(double nu, const SpdMatrix &sumsq_inv, bool inv) {
    return rWish_mt(GlobalRng::rng, nu, sumsq_inv, inv);
  }

  SpdMatrix rWish_mt(RNG &rng, double nu, const SpdMatrix &sumsq_inv,
                     bool inv) {
    uint d = sumsq_inv.nrow();
    Matrix L = WishartTriangle(rng, d, nu);
    bool ok = true;
    Matrix ss_chol = sumsq_inv.chol(ok);
    if (!ok) {
      report_error("problem in rWish");
    }

    Matrix tmp(ss_chol * L);  // tmp is the lower cholesky triangle of siginv
    if (inv) {
      report_error("need to invert from choelsky factor in rwish");
    }
    return LLT(tmp);
  }

  SpdMatrix rWishChol(double nu, const Matrix &sumsq_upper_chol, bool inv) {
    return rWishChol_mt(GlobalRng::rng, nu, sumsq_upper_chol, inv);
  }

  // sumsq_chol is the cholesky decomposition of the centered sum of
  // squares matrix, as in the sufficient statistic to the
  // multivariate normal model.  if(inv) then daw from the inverse
  // Wishart distribution (in the Bayesian world this ususally means
  // draw "Sigma"), otherwise draw from the ordinary Wishart
  // distribution (i.e. draw Sigma inverse).
  SpdMatrix rWishChol_mt(RNG &rng, double nu, const Matrix &sumsq_upper_chol,
                         bool inv) {
    uint d = sumsq_upper_chol.nrow();
    Matrix L = WishartTriangle(rng, d, nu);

    // sumsq_upper_chol = U (an upper triangular matrix)
    // if we're drawing sigma^{-1} then we want U^{-1}L times its transpose.
    // if we're drawing sigma then we want (U^T * L.inv) times (L.inv() U)
    SpdMatrix ans(L.nrow(), 0.0);
    const Matrix &U(sumsq_upper_chol);
    if (inv) {
      ans.add_inner(Lsolve(L, U));  // (L^{-1}U) (L^{-1}U)^T
    } else {
      L = ans.add_outer(Usolve(U, L));  // (U^{-1} L)^T U^{-1} L
    }
    return (ans);
  }

  // returns the density of the Wishart distribution evaluatated at
  // 'Siginv' with 'df' degrees of freedom and scale matrix sumsq (the
  // sufficient statistic for the variance in a multivariate normal
  // with known mean).
  //
  // I.e. sumsq = \sum_i (x_i-\mu)(x_i-\mu)^T + prior SS
  //
  // if(inv) then the density of the inverse Wishart is returned
  // instead.
  double dWish(const SpdMatrix &Siginv, const SpdMatrix &sumsq, double df,
               bool logscale, bool inv) {
    if (Siginv.nrow() != sumsq.nrow()) {
      report_error("Siginv and sumsq must have same dimensions in dWish");
    }

    const double log2 = 0.693147180559945;
    const double logpi = 1.1447298858494;

    int k = Siginv.nrow();
    double Sld(Siginv.logdet());
    double ssld(sumsq.logdet());
    double exponent = inv ? df + k + 1 : df - k - 1;
    double ans = -traceAB(Siginv, sumsq) + Sld * exponent + ssld * df;

    for (int i = 1; i <= k; ++i) ans -= lgamma((df + 1 - i));
    ans -= logpi * k * (k - 1) / 2.0;
    ans -= log2 * df * k;
    ans /= 2.0;
    return logscale ? ans : exp(ans);
  }

}  // namespace BOOM

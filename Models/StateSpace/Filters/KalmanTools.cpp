// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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

#include "Models/StateSpace/Filters/KalmanTools.hpp"
#include "distributions.hpp"
namespace BOOM {

  double scalar_kalman_update(double y, Vector &a, SpdMatrix &P, Vector &K,
                              double &F, double &v, bool missing,
                              const Vector &Z, double H, const Matrix &T,
                              Matrix &L, const SpdMatrix &RQR) {
    F = P.Mdist(Z) + H;
    double ans = 0;
    if (!missing) {
      K = T * (P * Z);
      K /= F;
      double mu = Z.dot(a);
      v = y - mu;
      ans = dnorm(y, mu, sqrt(F), true);
    } else {
      K = Z * 0;
      v = 0;
    }

    a = T * a;
    a += K * v;

    L = T.transpose();
    L.add_outer(Z, K, -1);  // L is the transpose of Durbin and Koopman's L
    P = T * P * L + RQR;

    return ans;
  }

  double vector_kalman_update(const Vector &y,
                              Vector &a,
                              SpdMatrix &P,
                              Matrix &K,
                              SpdMatrix &F,
                              Vector &v,
                              const Selector &observed,
                              Matrix Z,
                              SpdMatrix H,
                              const Matrix &T,
                              Matrix &L,
                              const SpdMatrix &RQR) {
    Vector Y = observed.select(y);
    Z = observed.select_rows(Z);
    H = observed.select(H);

    v = Y - Z * a;
    F = Z * P * Z.transpose() + H;
    SpdMatrix Finv = F.inv();
    K = T * P * Z.transpose() * Finv;
    Vector a_contemp = a + P * Z.Tmult(Finv * v);
    a = T * a_contemp;

    SpdMatrix Pcontemp = P - P * (Z.Tmult(Finv * Z)) * P;
    P = T * Pcontemp * T.transpose() + RQR;

    L = T - K * Z;

    return dmvn(v, Vector(v.size(), 0.0), Finv, Finv.logdet(), true);
  }

  void make_contemporaneous(Vector &a, SpdMatrix &P, double F, double v,
                            const Vector &Z) {
    Vector M = P * Z;
    a += M * (v / F);
    P.add_outer(M, -1.0 / F);
  }

  void scalar_kalman_smoother_update(Vector &a, SpdMatrix &P, const Vector &K,
                                     double F, double v, const Vector &Z,
                                     const Matrix &T, Vector &r, Matrix &N,
                                     Matrix &L) {
    L = T.transpose();
    L.add_outer(Z, K, -1);  // L is the transpose of Durbin and Koopman's L
    r = L * r + Z * (v / F);
    N = sandwich(L, N);
    a += P * r;
    P -= sandwich(P, N);
  }

}  // namespace BOOM

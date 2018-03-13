// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "Models/HMM/hmm_tools.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "uint.hpp"

namespace BOOM {
  using BOOM::uint;

  double fwd_1(Vector &pi, Matrix &P, const Matrix &logQ, const Vector &logd,
               const Vector &one) {
    /*----------------------------------------------------------------------
     * Input: pi[0..S-1] is the conditionl distribution of h[t-1]|Y[t-1]
     *        one[0..S-1] is a vector of 1's
     *        logd[s] is logp(y[t] | model[s])
     *        logQ[0..S-1] is the log of the square transition
     *                     probability matrix (rows of Q sum to 1)
     *
     * Output:  pi[0..S-1]  is prob(h[t] | Y[t])
     *          P[0..S-1]^2 is prob(h[t-1], h[t] | Y[t])
     *
     * Return:  logp(y[t] | y[1]..y[t-1])
     * --------------------------------------------------------------------*/
    uint S = pi.size();
    P = logQ;
    pi = log(pi);
    for (uint r = 0; r < S; ++r) P.row(r) += logd;  // P(r,s) += logd[s]
    for (uint s = 0; s < S; ++s) P.col(s) += pi;    // P(r,s) += pi[r]
    double m = max(P);
    P -= m;
    P.exp();
    double nc = P.abs_norm();
    P /= nc;
    pi = one * P;
    return m + log(nc);
  }

  void bkwd_1(Vector &pi, Matrix &P, Vector &wsp, const Vector &one) {
    /*----------------------------------------------------------------------
     * Input:  pi[0..S-1]  is prob(h[t]|Y[n])
     *         P[0..S-1]^2 is prob(h[t-1],h[t] | Y[t])
     *         wsp[0..S-1] is work space
     *         one[0..S-1] is a vector of 1's
     *
     * Output  pi is prob(h[t-1] | Y[n])
     *         P is prob(h[t-1],h[t] | Y[n])
     *----------------------------------------------------------------------*/
    wsp = pi / (one * P);  // ratio of new pi to old pi
    uint S = pi.size();
    for (uint r = 0; r < S; ++r) P.row(r) *= wsp;
    pi = P * one;
  }

}  // namespace BOOM

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
#include "LinAlg/Vector.hpp"

#include "cpputil/math_utils.hpp"

#include "distributions.hpp"
#include "numopt.hpp"

// Shamelessly adapted from R by Steven Scott.  The original comment
// below mentions an argument 'trace', which I removed.

namespace BOOM {
  const double E1 = 1.7182818; /* exp(1.0)-1.0 */
  const double big = 1.0e+35;  /*a very large number*/

  double simulated_annealing(Vector &pb, const Target &target, int maxit,
                             int tmax, double ti) {
    /* Given a starting point pb[0..n-1], simulated annealing
       minimization is performed on the function fminfn. The starting
       temperature is input as ti. To make sann work silently set
       trace to zero.  sann makes in total maxit function evaluations,
       tmax evaluations at each temperature. Returned quantities are
       pb (the location of the minimum), and yb (the minimum value of
    */
    long i, j;
    int k, its, itdoc;
    double t, y, dy, ytry, scale;
    //  double *p, *dp, *ptry;

    int n = pb.size();
    Vector p(n);
    Vector dp(n);
    Vector ptry(n);
    double yb = target(pb);
    if (!std::isfinite(yb)) yb = big;

    for (j = 0; j < n; j++) p[j] = pb[j];
    y = yb; /* init system state p, y */
    scale = 1.0 / ti;
    its = itdoc = 1;
    while (its < maxit) {             /* cool down system */
      t = ti / log((double)its + E1); /* temperature annealing schedule */
      k = 1;
      while ((k <= tmax) && (its < maxit)) /* iterate at constant temperature */
      {
        for (i = 0; i < n; i++)
          dp[i] = scale * t * rnorm(0, 1); /* random perturbation */
        for (i = 0; i < n; i++)
          ptry[i] = p[i] + dp[i]; /* new candidate point */
        ytry = target(ptry);      // fminfn (n, ptry, ex);
        if (!std::isfinite(ytry)) ytry = big;
        dy = ytry - y;
        if ((dy <= 0.0) ||
            (runif(0, 1) < exp(-dy / t))) { /* accept new point? */
          for (j = 0; j < n; j++) p[j] = ptry[j];
          y = ytry;    /* update system state p, y */
          if (y <= yb) /* if system state is best, then update best system state
                          pb, yb */
          {
            for (j = 0; j < n; j++) pb[j] = p[j];
            yb = y;
          }
        }
        its++;
        k++;
      }
      itdoc++;
    }
    return yb;
  }
}  // namespace BOOM

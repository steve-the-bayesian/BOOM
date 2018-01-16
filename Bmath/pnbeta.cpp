/*
  Copyright (C) 2005-2009 Steven L. Scott

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

/*
 *  Copyright (C) 2000-2008 The R Development Core Team
 *
 *  Algorithm AS 226 Appl. Statist. (1987) Vol. 36, No. 2
 *  Incorporates modification AS R84 from AS Vol. 39, pp311-2, 1990
 *                        and AS R95 from AS Vol. 44, pp551-2, 1995
 *  original (C) Royal Statistical Society 1987, 1990, 1995
 *
 *  Returns the cumulative probability of x for the non-central
 *  beta distribution with parameters a, b and non-centrality ncp.
 *
 *  Auxiliary routines required:
 *      lgamma - log-gamma function
 *      pbeta  - incomplete-beta function {nowadays: pbeta_raw() -> bratio()}
 */

#include "nmath.hpp"
#include "dpq.hpp"

namespace Rmath{
long double
pnbeta_raw(double x, double o_x, double a, double b, double ncp)
{
    /* o_x  == 1 - x  but maybe more accurate */

    /* change errmax and itrmax if desired;
     * original (AS 226, R84) had  (errmax; itrmax) = (1e-6; 100) */
    constexpr double errmax = 1.0e-9;
    const int    itrmax = 10000;  /* 100 is not enough for pf(ncp=200)
                                     see PR#11277 */

    double a0, ax, lbeta, c, errbd, temp, x0, tmp_c;
    int j, ierr;

    long double ans, gx, q, sumq;

    if (ncp < 0. || a <= 0. || b <= 0.) ML_ERR_return_NAN;

    if(x < 0. || o_x > 1. || (x == 0. && o_x == 1.)) return 0.;
    if(x > 1. || o_x < 0. || (x == 1. && o_x == 0.)) return 1.;

    c = ncp / 2.;

        /* initialize the series */

    x0 = floor(std::max<double>(c - 7. * sqrt(c), 0.));
    a0 = a + x0;
    lbeta = lgammafn(a0) + lgammafn(b) - lgammafn(a0 + b);
    /* temp = pbeta_raw(x, a0, b, true, false), but using (x, o_x): */
    bratio(a0, b, x, o_x, &temp, &tmp_c, &ierr, false);

    gx = exp(a0 * log(x) + b * (x < .5 ? log1p(-x) : log(o_x))
             - lbeta - log(a0));
    if (a0 > a)
        q = exp(-c + x0 * log(c) - lgammafn(x0 + 1.));
    else
        q = exp(-c);

    sumq = 1. - q;
    ans = ax = q * temp;

        /* recurse over subsequent terms until convergence is achieved */
    j = x0;
    do {
        j++;
        temp -= gx;
        gx *= x * (a + b + j - 1.) / (a + j);
        q *= c / j;
        sumq -= q;
        ax = temp * q;
        ans += ax;
        errbd = (temp - gx) * sumq;
    }
    while (errbd > errmax && j < itrmax + x0);

    if (errbd > errmax){
     report_error("full precision was not achieved in pnbeta");
    }
    if (j >= itrmax + x0){
     report_error("algorithm did not converge in pnbeta");
    }

    return ans;
}

double pnbeta2(double x, double o_x, double a, double b, double ncp,
        /* o_x  == 1 - x  but maybe more accurate */
        int lower_tail, int log_p)
{
    long double ans= pnbeta_raw(x, o_x, a,b, ncp);

    /* return R_DT_val(ans), but we want to warn about cancellation here */
    if(lower_tail) return log_p ? log(ans) : ans;
    else {
      if(ans > 1 - 1e-10){
       report_error("full precision was not achieved in pnbeta");
      }
      ans = std::min<double>(ans, 1.0);  /* Precaution */
      return log_p ? log1p(-ans) : (1 - ans);
    }
}

double pnbeta(double x, double a, double b, double ncp,
              int lower_tail, int log_p)
{
    if (isnan(x) || isnan(a) || isnan(b) || isnan(ncp))
        return x + a + b + ncp;

    R_P_bounds_01(x, 0., 1.);
    return pnbeta2(x, 1-x, a, b, ncp, lower_tail, log_p);
}

}

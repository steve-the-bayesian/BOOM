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

/*
 *  Algorithm AS 275 Appl.Statist. (1992), vol.41, no.2
 *  original  (C) 1992       Royal Statistical Society
 *  Copyright (C) 2000--2002 The R Development Core Team
 *  Copyright (C) 2003       The R Foundation
 *
 *  Computes the noncentral chi-squared distribution function with
 *  positive real degrees of freedom f and nonnegative noncentrality
 *  parameter theta
 */

#include "nmath.hpp"
#include "dpq.hpp"
#include <stdexcept>
#include <sstream>

namespace Rmath{

/*----------- DEBUGGING -------------
 *
 *      make CFLAGS='-DDEBUG_pnch ....'

 * -- Feb.6, 2000 (R pre0.99); M.Maechler:  still have
 * bad precision & non-convergence in some cases (x ~= f, both LARGE)
 */

double pnchisq(double x, double f, double theta, int lower_tail, int log_p)
{
#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(f) || ISNAN(theta))
        return x + f + theta;
    if (!R_FINITE(f) || !R_FINITE(theta))
        ML_ERR_return_NAN;
#endif

    if (f < 0. || theta < 0.) ML_ERR_return_NAN;

    return (R_DT_val(pnchisq_raw(x, f, theta, 1e-12, 10000)));
}

double pnchisq_raw(double x, double f, double theta,
                   double errmax, int itrmax)
{
    double ans, lam, u, v, x2, f2, t, term, bound, f_x_2n, f_2n;
    int n, flag;

    const double my_dbl_min_exp =
        M_LN2 * std::numeric_limits<double>::min_exponent;
    /*= -708.3964 for IEEE double precision */

    if (x <= 0.)        return 0.;
    if(!R_FINITE(x))    return 1.;

    lam = .5 * theta;
    if(-lam < my_dbl_min_exp){
      std::ostringstream err;
      err << "non centrality parameter (=" << theta
          << ") too large for current algorithm" << std::endl;
      report_error(err.str());
    }
    /* evaluate the first term */

    v = u = exp(-lam);
    x2 = .5 * x;
    f2 = .5 * f;
    f_x_2n = f - x;


    if(f2 * std::numeric_limits<double>::epsilon() > 0.125 &&
       fabs(t = x2 - f2) < sqrt(std::numeric_limits<double>::epsilon()) * f2) {
        /* evade cancellation error */
        t = exp((1 - t)*(2 - t/(f2 + 1))) / sqrt(2*M_PI*(f2 + 1));
    }
    else {
        /* careful not to overflow .. : */
        t = f2*log(x2) -x2 - lgammafn(f2 + 1);
        if (t < my_dbl_min_exp &&
            x > f + theta +  3* sqrt( 2*(f + 2*theta))) {
            /* x > E[X] + 3* sigma(X) */
            return 1.; /* better than 0 ! */
        } /* else */
        t = exp(t);
    }

    if(t <= 0) {
      std::ostringstream err;
      err << "too large x (=" << theta << ")"
          << " or centrality parameter " << x
          << " for current algorithm.  Result is probably invalid!";
      report_error(err.str());
    }

    term = v * t;
    ans = term;

    /* check if (f+2n) is greater than x */

    flag = false;
    n = 1;
    f_2n = f + 2.;/* = f + 2*n */
    f_x_2n += 2.;/* = f - x + 2*n */
    for(;;) {
        if (f_x_2n > 0) {

            /* find the error bound and check for convergence */
            flag = true;
            goto L10;
        }
        for(;;) {
            /* evaluate the next term of the */
            /* expansion and then the partial sum */

            u *= lam / n;
            v += u;
            t *= x / f_2n;
            term = v * t;
            ans += term;
            n++; f_2n += 2; f_x_2n += 2;
            if (!flag && n <= itrmax)
                break;
        L10:
            bound = t * x / f_x_2n;
            if (bound <= errmax || n > itrmax)
                goto L_End;
        }
    }
L_End:
    if (bound > errmax) { /* NOT converged */
        ML_ERROR(ME_PRECISION);
    }
    return (ans);
}
}

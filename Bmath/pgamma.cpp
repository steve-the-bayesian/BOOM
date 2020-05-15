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
 *  Mathlib : A C Library of Special Functions
 *  Copyright (C) 1998          Ross Ihaka
 *  Copyright (C) 1999-2000     The R Development Core Team
 *  based on AS 239 (C) 1988 Royal Statistical Society
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA.
 *
 *  SYNOPSIS
 *
 *      #include "Bmath.hpp"
 *      double pgamma(double x, double alph, double scale,
 *                    int lower_tail, int log_p)
 *
 *  DESCRIPTION
 *
 *      This function computes the distribution function for the
 *      gamma distribution with shape parameter alph and scale parameter
 *      scale.  This is also known as the incomplete gamma function.
 *      See Abramowitz and Stegun (6.5.1) for example.
 *
 *  NOTES
 *
 *      This function is an adaptation of Algorithm 239 from the
 *      Applied Statistics Series.  The algorithm is faster than
 *      those by W. Fullerton in the FNLIB library and also the
 *      TOMS 542 alorithm of W. Gautschi.  It provides comparable
 *      accuracy to those algorithms and is considerably simpler.
 *
 *  REFERENCES
 *
 *      Algorithm AS 239, Incomplete Gamma Function
 *      Applied Statistics 37, 1988.
 */

#include "nmath.hpp"
#include "dpq.hpp"
namespace Rmath{
/*----------- DEBUGGING -------------
 *      make CFLAGS='-DDEBUG_p -g -I/usr/local/include -I../include'
 */


double pgamma(double x, double alph, double scale, int lower_tail, int log_p) {
  const double xbig = 1.0e+8;
  const double xlarge = 1.0e+37;
  const double alphlimit = 1000.;  // normal approx. for alph > alphlimit

#ifndef IEEE_754
    /* was elimit = -88.0e0; */
#endif

    double pn1, pn2, pn3, pn4, pn5, pn6, arg, a, b, c, an, osum, sum;
    long n;
    int pearson;

    /* check that we have valid values for x and alph */

#ifdef IEEE_754
    if (ISNAN(x) || ISNAN(alph) || ISNAN(scale))
        return x + alph + scale;
#endif
    if(alph <= 0. || scale <= 0.)
        ML_ERR_return_NAN;

    x /= scale;
#ifdef IEEE_754
    if (ISNAN(x)) /* eg. original x = scale = Inf */
        return x;
#endif
    if (x <= 0.)
        return R_DT_0;

    /* use a normal approximation if alph > alphlimit */

    if (alph > alphlimit) {
        pn1 = sqrt(alph) * 3. * (pow(x/alph, 1./3.) + 1. / (9. * alph) - 1.);
        return pnorm(pn1, 0., 1., lower_tail, log_p);
    }

    /* if x is extremely large __compared to alph__ then return 1 */

    if (x > xbig * alph)
        return R_DT_1;

    if (x <= 1. || x < alph) {

        pearson = 1;/* use pearson's series expansion. */

        arg = alph * log(x) - x - lgammafn(alph + 1.);
        c = 1.;
        sum = 1.;
        a = alph;
        do {
            a += 1.;
            c *= x / a;
            sum += c;
        } while (c > std::numeric_limits<double>::epsilon());
        arg += log(sum);
    }
    else { /* x >= max( 1, alph) */

        pearson = 0;/* use a continued fraction expansion */

        arg = alph * log(x) - x - lgammafn(alph);
        a = 1. - alph;
        b = a + x + 1.;
        pn1 = 1.;
        pn2 = x;
        pn3 = x + 1.;
        pn4 = x * b;
        sum = pn3 / pn4;
        for (n = 1; ; n++) {
            a += 1.;/* =   n+1 -alph */
            b += 2.;/* = 2(n+1)-alph+x */
            an = a * n;
            pn5 = b * pn3 - an * pn1;
            pn6 = b * pn4 - an * pn2;
            if (fabs(pn6) > 0.) {
                osum = sum;
                sum = pn5 / pn6;
                if (fabs(osum - sum) <=
                    std::numeric_limits<double>::epsilon() * std::min(1., sum))
                  break;
            }
            pn1 = pn3;
            pn2 = pn4;
            pn3 = pn5;
            pn4 = pn6;
            if (fabs(pn5) >= xlarge) {

                /* re-scale the terms in continued fraction if they are large */
                pn1 /= xlarge;
                pn2 /= xlarge;
                pn3 /= xlarge;
                pn4 /= xlarge;
            }
        }
        arg += log(sum);
    }

    lower_tail = (lower_tail == pearson);

    if (log_p && lower_tail)
        return(arg);
    /* else */
#ifndef IEEE_754
    /* Underflow check :*/
    // set exp(E) = 0 for E < elimit !
    const double elimit = M_LN2*(std::numeric_limits<double>::min_exponent);
    if (arg < elimit)
        sum = 0.;
    else
#endif
        sum = exp(arg);

    return (lower_tail) ? sum : R_D_val(1 - sum);
}
}  // namespace Rmath

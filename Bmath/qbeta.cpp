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
 *  R : A Computer Language for Statistical Data Analysis
 *  Copyright (C) 1995, 1996  Robert Gentleman and Ross Ihaka
 *  Copyright (C) 1998--2007  The R Development Core Team
 *  based on code (C) 1979 and later Royal Statistical Society
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
 *  along with this program; if not, a copy is available at
 *  http://www.r-project.org/Licenses/
 *

 * Reference:
 * Cran, G. W., K. J. Martin and G. E. Thomas (1977).
 *      Remark AS R19 and Algorithm AS 109,
 *      Applied Statistics, 26(1), 111-114.
 * Remark AS R83 (v.39, 309-310) and the correction (v.40(1) p.236)
 *      have been incorporated in this version.
 */

#include "nmath.hpp"
#include "dpq.hpp"
namespace Rmath{

/* set the exponent of accu to -2r-2 for r digits of accuracy */
/*---- NEW ---- -- still fails for p = 1e11, q=.5*/

const double fpu(3e-308);
/* acu_min:  Minimal value for accuracy 'acu' which will depend on (a,p);
             acu_min >= fpu ! */

const double acu_min (1e-300);

const double lower(fpu);
const double upper(1-2.22e-16);

const double const1(2.30753);
const double const2(0.27061);
const double const3(0.99229);
const double const4(0.04481);


double qbeta(double alpha, double p, double q, int lower_tail, int log_p)
{
    int swap_tail, i_pb, i_inn;
    double a, adj, logbeta, g, h, pp, p_, prev, qq, r, s, t, tx, w, y, yprev;
    double acu;
    volatile double xinbta;

    /* test for admissibility of parameters */

    if (isnan(p) || isnan(q) || isnan(alpha)){
        return p + q + alpha;
    }

    if(p < 0. || q < 0.){
      report_error("shape parameters for qbeta must be > 0.");
    }

    R_Q_P01_boundaries(alpha, 0, 1);

    p_ = R_DT_qIv(alpha);/* lower_tail prob (in any case) */

    if(log_p && (p_ == 0. || p_ == 1.))
        return p_; /* better than NaN or infinite loop;
                      FIXME: suboptimal, since -Inf < alpha ! */

    /* initialize */
    logbeta = lbeta(p, q);

    /* change tail if necessary;  afterwards   0 < a <= 1/2      */
    if (p_ <= 0.5) {
        a = p_; pp = p; qq = q; swap_tail = 0;
    } else { /* change tail, swap  p <-> q :*/
        a = (!lower_tail && !log_p)? alpha : 1 - p_;
        pp = q; qq = p; swap_tail = 1;
    }

    /* calculate the initial approximation */

    /* y := {fast approximation of} qnorm(1 - a) :*/
    r = sqrt(-2 * log(a));
    y = r - (const1 + const2 * r) / (1. + (const3 + const4 * r) * r);
    if (pp > 1 && qq > 1) {
        r = (y * y - 3.) / 6.;
        s = 1. / (pp + pp - 1.);
        t = 1. / (qq + qq - 1.);
        h = 2. / (s + t);
        w = y * sqrt(h + r) / h - (t - s) * (r + 5. / 6. - 2. / (3. * h));
        xinbta = pp / (pp + qq * exp(w + w));
    } else {
        r = qq + qq;
        t = 1. / (9. * qq);
        t = r * pow(1. - t + y * sqrt(t), 3.0);
        if (t <= 0.)
            xinbta = 1. - exp((log1p(-a)+ log(qq) + logbeta) / qq);
        else {
            t = (4. * pp + r - 2.) / t;
            if (t <= 1.)
                xinbta = exp((log(a * pp) + logbeta) / pp);
            else
                xinbta = 1. - 2. / (t + 1.);
        }
    }

    /* solve for x by a modified newton-raphson method, */
    /* using the function pbeta_raw */

    r = 1 - pp;
    t = 1 - qq;
    yprev = 0.;
    adj = 1;
    /* Sometimes the approximation is negative! */
    if (xinbta < lower)
        xinbta = 0.5;
    else if (xinbta > upper)
        xinbta = 0.5;

    /* Desired accuracy should depend on  (a,p)
     * This is from Remark .. on AS 109, adapted.
     * However, it's not clear if this is "optimal" for IEEE double prec.

     * acu = std::max<double>(acu_min, pow(10., -25. - 5./(pp * pp) - 1./(a * a)));

     * NEW: 'acu' accuracy NOT for squared adjustment, but simple;
     * ---- i.e.,  "new acu" = sqrt(old acu)

    */
    acu = std::max<double>(acu_min, pow(10., -13 - 2.5/(pp * pp) - 0.5/(a * a)));
    tx = prev = 0.;     /* keep -Wall happy */

    for (i_pb=0; i_pb < 1000; i_pb++) {
        y = pbeta_raw(xinbta, pp, qq, /*lower_tail = */ true, false);
        if(!std::isfinite(y)){
          report_error("algorithm blew up ni qbeta");
        }

        y = (y - a) *
            exp(logbeta + r * log(xinbta) + t * log1p(-xinbta));
        if (y * yprev <= 0.)
            prev = std::max<double>(fabs(adj),fpu);
        g = 1;
        for (i_inn=0; i_inn < 1000;i_inn++) {
            adj = g * y;
            if (fabs(adj) < prev) {
                tx = xinbta - adj; /* trial new x */
                if (tx >= 0. && tx <= 1) {
                    if (prev <= acu)    goto L_converged;
                    if (fabs(y) <= acu) goto L_converged;
                    if (tx != 0. && tx != 1)
                        break;
                }
            }
            g /= 3;
        }
        if (fabs(tx - xinbta) < 1e-15*xinbta) goto L_converged;
        xinbta = tx;
        yprev = y;
    }
    /*-- NOT converged: Iteration count --*/
    report_error("algorithm did not converge in qbeta");

L_converged:
    return swap_tail ? 1 - xinbta : xinbta;
}



}

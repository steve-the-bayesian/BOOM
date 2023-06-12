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
 *  Copyright (C) 1998-2001 Ross Ihaka and the R Development Core Team
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
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif
#include "nmath.hpp"
namespace Rmath{

#ifndef IEEE_754

void ml_error(int n)
{
    switch(n) {

    case ME_NONE:
        errno = 0;
        break;

    case ME_DOMAIN:
      report_error("Bmath domain error");
      break;
    case ME_NOCONV:
      report_error("failed to converge");
      break;

    case ME_RANGE:
      report_error("Bmath range error");
      break;

    default:
      report_error("call to Bmath::ml_error with unknown error");
      break;
    }
}

#endif

#ifdef MATHLIB_STANDALONE
/*
 *  based on code in ../main/arithmetic.c
 */


#ifdef IEEE_754

int R_IsNaNorNA(double x)
{
/* NOTE: some systems do not return 1 for true. */
    return (isnan(x) != 0);
}

/* Include the header file defining finite() */
#ifdef HAVE_IEEE754_H
# include <ieee754.h>           /* newer Linuxen */
#else
# ifdef HAVE_IEEEFP_H
#  include <ieeefp.h>           /* others [Solaris 2.5.x], .. */
# endif
#endif
#if defined(Win32) && defined(_MSC_VER)
# include <float.h>
#endif

/* int R_finite(double x) */
/* { */
/* #ifdef Macintosh */
/*     return isfinite(x); */
/* #endif */
/* #ifdef HAVE_WORKING_FINITE */
/*     return finite(x); */
/* #else */
/* # ifdef _AIX */
/* #  include <fp.h> */
/*      return FINITE(x); */
/* # else */
/*     return (!isnan(x) & (x != BOOM::infinity()) & (x != BOOM::negative_infinity())); */
/* # endif */
/* #endif */
/* } */

#else /* not IEEE_754 */

int R_IsNaNorNA(double x)
{
# ifndef HAVE_ISNAN
  return (x == std::numeric_limits<double>::quiet_NaN());
# else
  return (isnan(x) != 0 || x == std::numeric_limits<double>::quiet_NaN());
# endif
}

/* bool R_finite(double x) */
/* { */
/* # ifndef HAVE_FINITE */
/*     return (x !=  numeric_limits<double>::quiet_NaN() && x < BOOM::infinity() && x > BOOM::negative_infinity()); */
/* # else */
/*     int finite(double); */
/*     return finite(x); */
/* # endif */
/* } */
#endif /* IEEE_754 */

inline double myfmod(double x1, double x2)
{
    double q = x1 / x2;
    return x1 - FLOOR(q) * x2;
}

#ifdef HAVE_WORKING_LOG
# define R_log  log
#else
double R_log(double x) {
  return (x > 0 ? log(x)
                : x < 0 ? std::numeric_limits<double>::quiet_NaN()
                        : BOOM::negative_infinity());
}
#endif

double R_pow(double x, double y) /* = x ^ y */
{
    if (x == 1. || y == 0.)
        return(1.);
    if (x == 0.) {
      if (y > 0.) {
        return(0.);
      } else {
        /* y <= 0 */
        return(BOOM::infinity());
      }
    }
    if (R_FINITE(x) && R_FINITE(y))
        return(pow(x,y));
    if (ISNAN(x) || ISNAN(y)) {
#ifdef IEEE_754
        return(x + y);
#else
        return std::numeric_limits<double>::quiet_NaN();
#endif
    }
    if (!R_FINITE(x)) {
        if (x > 0)               /* Inf ^ y */
            return((y < 0.)? 0. : BOOM::infinity());
        else {                  /* (-Inf) ^ y */
            if (R_FINITE(y) && y == FLOOR(y)) /* (-Inf) ^ n */
                return((y < 0.) ? 0. : (myfmod(y,2.) ? x  : -x));
        }
    }
    if (!R_FINITE(y)) {
        if (x >= 0) {
            if (y > 0)           /* y == +Inf */
                return((x >= 1)? BOOM::infinity() : 0.);
            else                /* y == -Inf */
                return((x < 1) ? BOOM::infinity() : 0.);
        }
    }
    return (std::numeric_limits<double>::quiet_NaN()); /* all other cases:
                     (-Inf)^{+-Inf,
                     non-int}; (neg)^{+-Inf} */
}

double R_pow_di(double x, int n)
{
    double pow = 1.0;

    if (ISNAN(x)) return x;
    if (n != 0) {
        if (!R_FINITE(x)) return R_pow(x, (double)n);
        if (n < 0) { n = -n; x = 1/x; }
        for(;;) {
            if (n & 01) pow *= x;
            if (n >>= 1) x *= x; else break;
        }
    }
    return pow;
}

double NA_REAL = std::numeric_limits<double>::quiet_NaN();
double R_PosInf = BOOM::infinity(), R_NegInf = BOOM::negative_infinity();

#endif /* MATHLIB_STANDALONE */
}

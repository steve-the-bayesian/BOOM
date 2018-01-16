/*
  Copyright (C) 2005-2013 Steven L. Scott

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

// This file was modified from the public domain cephes math library
// taken from netlib.
#include "cephes_impl.hpp"

namespace BOOM {
  namespace Cephes {

  /*                                            rgamma.c
   *
   *    Reciprocal gamma function
   *
   *
   *
   * SYNOPSIS:
   *
   * double x, y, rgamma();
   *
   * y = rgamma( x );
   *
   *
   *
   * DESCRIPTION:
   *
   * Returns one divided by the gamma function of the argument.
   *
   * The function is approximated by a Chebyshev expansion in
   * the interval [0,1].  Range reduction is by recurrence
   * for arguments between -34.034 and +34.84425627277176174.
   * 1/MAXNUM is returned for positive arguments outside this
   * range.  For arguments less than -34.034 the cosecant
   * reflection formula is applied; lograrithms are employed
   * to avoid unnecessary overflow.
   *
   * The reciprocal gamma function has no singularities,
   * but overflow and underflow may occur for large arguments.
   * These conditions return either MAXNUM or 1/MAXNUM with
   * appropriate sign.
   *
   * ACCURACY:
   *
   *                      Relative error:
   * arithmetic   domain     # trials      peak         rms
   *    DEC      -30,+30       4000       1.2e-16     1.8e-17
   *    IEEE     -30,+30      30000       1.1e-15     2.0e-16
   * For arguments less than -34.034 the peak error is on the
   * order of 5e-15 (DEC), excepting overflow or underflow.
   */

  /*
    Cephes Math Library Release 2.8:  June, 2000
    Copyright 1985, 1987, 2000 by Stephen L. Moshier
  */

  /* Chebyshev coefficients for reciprocal gamma function
   * in interval 0 to 1.  Function is 1/(x gamma(x)) - 1
   */

  static double R[] = {
    3.13173458231230000000E-17,
    -6.70718606477908000000E-16,
    2.20039078172259550000E-15,
    2.47691630348254132600E-13,
    -6.60074100411295197440E-12,
    5.13850186324226978840E-11,
    1.08965386454418662084E-9,
    -3.33964630686836942556E-8,
    2.68975996440595483619E-7,
    2.96001177518801696639E-6,
    -8.04814124978471142852E-5,
    4.16609138709688864714E-4,
    5.06579864028608725080E-3,
    -6.41925436109158228810E-2,
    -4.98558728684003594785E-3,
    1.27546015610523951063E-1
  };

  double rgamma(double x) {
    double w, y, z;
    int sign;

    if( x > 34.84425627277176174)
    {
      report_error("Underflow error in BOOM::Cephes::rgamma.");
      return(1.0/MAXNUM);
    }
    if( x < -34.034 )
    {
      w = -x;
      z = std::sin( PI*w );
      if( z == 0.0 )
        return(0.0);
      if( z < 0.0 )
      {
        sign = 1;
        z = -z;
      }
      else
        sign = -1;

      y = log( w * z ) - log(PI) + ::lgamma(w);
      if( y < -MAXLOG )
      {
        report_error("Underflow error in BOOM::Cephes::rgamma");
        return( sign * 1.0 / MAXNUM );
      }
      if( y > MAXLOG )
      {
        report_error("Overflow error in BOOM::Cephes::rgamma");
        return( sign * MAXNUM );
      }
      return( sign * exp(y));
    }
    z = 1.0;
    w = x;

    while( w > 1.0 )    /* Downward recurrence */
    {
      w -= 1.0;
      z *= w;
    }
    while( w < 0.0 )    /* Upward recurrence */
    {
      z /= w;
      w += 1.0;
    }
    if( w == 0.0 )              /* Nonpositive integer */
      return(0.0);
    if( w == 1.0 )              /* Other integer */
      return( 1.0/z );

    y = w * ( 1.0 + chbevl( 4.0*w-2.0, R, 16 ) ) / z;
    return(y);
  }
  }  // namespace Cephes
}  // namespace BOOM

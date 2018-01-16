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

  /*                                                      powi.c
   *
   *      Real raised to integer power
   *
   *
   *
   * SYNOPSIS:
   *
   * double x, y, powi();
   * int n;
   *
   * y = powi( x, n );
   *
   *
   *
   * DESCRIPTION:
   *
   * Returns argument x raised to the nth power.
   * The routine efficiently decomposes n as a sum of powers of
   * two. The desired power is a product of two-to-the-kth
   * powers of x.  Thus to compute the 32767 power of x requires
   * 28 multiplications instead of 32767 multiplications.
   *
   *
   *
   * ACCURACY:
   *
   *
   *                      Relative error:
   * arithmetic   x domain   n domain  # trials      peak         rms
   *    DEC       .04,26     -26,26    100000       2.7e-16     4.3e-17
   *    IEEE      .04,26     -26,26     50000       2.0e-15     3.8e-16
   *    IEEE        1,2    -1022,1023   50000       8.6e-14     1.6e-14
   *
   * Returns MAXNUM on overflow, zero on underflow.
   *
   */

  /*                                                      powi.c  */

  /*
    Cephes Math Library Release 2.3:  March, 1995
    Copyright 1984, 1995 by Stephen L. Moshier
  */

  double powi(double x, int nn) {
    int n, e, sign, asign, lx;
    double w, y, s;

    /* See pow.c for these tests.  */
    if( x == 0.0 )
    {
      if( nn == 0 )
        return( 1.0 );
      else if( nn < 0 )
        return(-1 * std::numeric_limits<double>::infinity());
      else
      {
        if( nn & 1 )
          return( x );
        else
          return( 0.0 );
      }
    }

    if( nn == 0 )
      return( 1.0 );

    if( nn == -1 )
      return( 1.0/x );

    if( x < 0.0 )
    {
      asign = -1;
      x = -x;
    }
    else
      asign = 0;


    if( nn < 0 )
    {
      sign = -1;
      n = -nn;
    }
    else
    {
      sign = 1;
      n = nn;
    }

    /* Even power will be positive. */
    if( (n & 1) == 0 )
      asign = 0;

    /* Overflow detection */

    /* Calculate approximate logarithm of answer */
    s = std::frexp( x, &lx );
    e = (lx - 1)*n;
    if( (e == 0) || (e > 64) || (e < -64) )
    {
      s = (s - 7.0710678118654752e-1) / (s +  7.0710678118654752e-1);
      s = (2.9142135623730950 * s - 0.5 + lx) * nn * LOGE2;
    }
    else
    {
      s = LOGE2 * e;
    }

    if( s > MAXLOG )
    {
      report_error("Overflow error in BOOM::Cephes::powi().");
      y = std::numeric_limits<double>::infinity();
      goto done;
    }

    /* do not produce denormal answer */
    if( s < -MAXLOG )
      return(0.0);

    /* First bit of the power */
    if( n & 1 )
      y = x;

    else
      y = 1.0;

    w = x;
    n >>= 1;
    while( n )
    {
      w = w * w;      /* arg to the 2-to-the-kth power */
      if( n & 1 )     /* if that bit is set, then include in product */
        y *= w;
      n >>= 1;
    }

    if( sign < 0 )
      y = 1.0/y;

 done:

    if( asign )
    {
      /* odd power of negative number */
      if( y == 0.0 )
        y = -0;
      else
        y = -y;
    }
    return(y);
  }
  }  // namespace Cephes
}  // namespace BOOM

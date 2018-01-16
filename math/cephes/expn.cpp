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
  /*                                                    expn.c
   *
   *            Exponential integral En
   *
   *
   *
   * SYNOPSIS:
   *
   * int n;
   * double x, y, expn();
   *
   * y = expn( n, x );
   *
   *
   *
   * DESCRIPTION:
   *
   * Evaluates the exponential integral
   *
   *                 inf.
   *                   -
   *                  | |   -xt
   *                  |    e
   *      E (x)  =    |    ----  dt.
   *       n          |      n
   *                | |     t
   *                 -
   *                  1
   *
   *
   * Both n and x must be nonnegative.
   *
   * The routine employs either a power series, a continued
   * fraction, or an asymptotic formula depending on the
   * relative values of n and x.
   *
   * ACCURACY:
   *
   *                      Relative error:
   * arithmetic   domain     # trials      peak         rms
   *    DEC       0, 30        5000       2.0e-16     4.6e-17
   *    IEEE      0, 30       10000       1.7e-15     3.6e-16
   *
   */

  /*                                                    expn.c  */

  /* Cephes Math Library Release 2.8:  June, 2000
     Copyright 1985, 2000 by Stephen L. Moshier */


  double expn(int n, double x) {
    double ans, r, t, yk, xk;
    double pk, pkm1, pkm2, qk, qkm1, qkm2;
    double psi, z;
    int i, k;
    const double big = 1.44115188075855872E+17;

    if( n < 0 )
      goto domerr;

    if( x < 0 )
    {
   domerr: report_error("Domain error in expn.");
      return( MAXNUM );
    }

    if( x > MAXLOG )
      return( 0.0 );

    if( x == 0.0 )
    {
      if( n < 2 )
      {
        report_error("Singularity in BOOM::Cephes::expn().");
        return( MAXNUM );
      }
      else
        return( 1.0/(n-1.0) );
    }

    if( n == 0 )
      return( exp(-x)/x );

    /*                                                  expn.c  */
    /*          Expansion for large n           */

    if( n > 5000 )
    {
      xk = x + n;
      yk = 1.0 / (xk * xk);
      t = n;
      ans = yk * t * (6.0 * x * x  -  8.0 * t * x  +  t * t);
      ans = yk * (ans + t * (t  -  2.0 * x));
      ans = yk * (ans + t);
      ans = (ans + 1.0) * exp( -x ) / xk;
      goto done;
    }

    if( x > 1.0 )
      goto cfrac;

    /*                                                  expn.c  */

    /*          Power series expansion          */

    psi = eulers_constant - log(x);
    for( i=1; i<n; i++ )
      psi = psi + 1.0/i;

    z = -x;
    xk = 0.0;
    yk = 1.0;
    pk = 1.0 - n;
    if( n == 1 )
      ans = 0.0;
    else
      ans = 1.0/pk;
    do
    {
      xk += 1.0;
      yk *= z/xk;
      pk += 1.0;
      if( pk != 0.0 )
      {
        ans += yk/pk;
      }
      if( ans != 0.0 )
        t = std::fabs(yk/ans);
      else
        t = 1.0;
    }
    while( t > MACHEP );
    k = xk;
    t = n;
    r = n - 1;
    ans = (std::pow(z, r) * psi / ::tgamma(t)) - ans;
    goto done;

    /*                                                  expn.c  */
    /*          continued fraction              */
 cfrac:
    k = 1;
    pkm2 = 1.0;
    qkm2 = x;
    pkm1 = 1.0;
    qkm1 = x + n;
    ans = pkm1/qkm1;

    do
    {
      k += 1;
      if( k & 1 )
      {
        yk = 1.0;
        xk = n + (k-1)/2;
      }
      else
      {
        yk = x;
        xk = k/2;
      }
      pk = pkm1 * yk  +  pkm2 * xk;
      qk = qkm1 * yk  +  qkm2 * xk;
      if( qk != 0 )
      {
        r = pk/qk;
        t = std::fabs( (ans - r)/r );
        ans = r;
      }
      else
        t = 1.0;
      pkm2 = pkm1;
      pkm1 = pk;
      qkm2 = qkm1;
      qkm1 = qk;
      if( std::fabs(pk) > big )
      {
        pkm2 /= big;
        pkm1 /= big;
        qkm2 /= big;
        qkm1 /= big;
      }
    }
    while( t > MACHEP );

    ans *= exp( -x );

 done:
    return( ans );
  }
  }  // namespace Cephes
}  // namespace BOOM

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

  /*                                                    fac.c
   *
   *    Factorial function
   *
   *
   *
   * SYNOPSIS:
   *
   * double y, fac();
   * int i;
   *
   * y = fac( i );
   *
   *
   *
   * DESCRIPTION:
   *
   * Returns factorial of i  =  1 * 2 * 3 * ... * i.
   * fac(0) = 1.0.
   *
   * Due to machine arithmetic bounds the largest value of
   * i accepted is 33 in DEC arithmetic or 170 in IEEE
   * arithmetic.  Greater values, or negative ones,
   * produce an error message and return MAXNUM.
   *
   *
   *
   * ACCURACY:
   *
   * For i < 34 the values are simply tabulated, and have
   * full machine accuracy.  If i > 55, fac(i) = gamma(i+1);
   * see gamma.c.
   *
   *                      Relative error:
   * arithmetic   domain      peak
   *    IEEE      0, 170    1.4e-15
   *    DEC       0, 33      1.4e-17
   *
   */

  /*
    Cephes Math Library Release 2.8:  June, 2000
    Copyright 1984, 1987, 2000 by Stephen L. Moshier
  */

  /* Factorials of integers from 0 through 33 */
  static double factbl[] = {
    1.00000000000000000000E0,
    1.00000000000000000000E0,
    2.00000000000000000000E0,
    6.00000000000000000000E0,
    2.40000000000000000000E1,
    1.20000000000000000000E2,
    7.20000000000000000000E2,
    5.04000000000000000000E3,
    4.03200000000000000000E4,
    3.62880000000000000000E5,
    3.62880000000000000000E6,
    3.99168000000000000000E7,
    4.79001600000000000000E8,
    6.22702080000000000000E9,
    8.71782912000000000000E10,
    1.30767436800000000000E12,
    2.09227898880000000000E13,
    3.55687428096000000000E14,
    6.40237370572800000000E15,
    1.21645100408832000000E17,
    2.43290200817664000000E18,
    5.10909421717094400000E19,
    1.12400072777760768000E21,
    2.58520167388849766400E22,
    6.20448401733239439360E23,
    1.55112100433309859840E25,
    4.03291461126605635584E26,
    1.0888869450418352160768E28,
    3.04888344611713860501504E29,
    8.841761993739701954543616E30,
    2.6525285981219105863630848E32,
    8.22283865417792281772556288E33,
    2.6313083693369353016721801216E35,
    8.68331761881188649551819440128E36
  };
    const int MAXFAC =  33;

    double fac(int i) {
      double f, n;
      int j;

      if( i < 0 ) {
        report_error("i < 0 in call to fac(i)");
        return MAXNUM;
      }

      if( i > MAXFAC ) {
        report_error("i > MAXFAC in call to fac(i).");
        return( MAXNUM );
      }

      /* Get answer from table for small i. */
      if( i < 34 ) {
        return( factbl[i] );
      }
      /* Use gamma function for large i. */
      if( i > 55 ) {
        double x = i + 1;
        return ::tgamma(x);
      }
      /* Compute directly for intermediate i. */
      n = 34.0;
      f = 34.0;
      for( j=35; j<=i; j++ ) {
        n += 1.0;
        f *= n;
      }
      f *= factbl[33];
      return( f );
    }
  }  // namespace Cephes
}  // namespace BOOM

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
  /*                                                    polylog.c
   *
   *    Polylogarithms
   *
   *
   *
   * SYNOPSIS:
   *
   * double x, y, polylog();
   * int n;
   *
   * y = polylog( n, x );
   *
   *
   * The polylogarithm of order n is defined by the series
   *
   *
   *              inf   k
   *               -   x
   *  Li (x)  =    >   ---  .
   *    n          -     n
   *              k=1   k
   *
   *
   *  For x = 1,
   *
   *               inf
   *                -    1
   *   Li (1)  =    >   ---   =  Riemann zeta function (n)  .
   *     n          -     n
   *               k=1   k
   *
   *
   *  When n = 2, the function is the dilogarithm, related to Spence's integral:
   *
   *                 x                      1-x
   *                 -                        -
   *                | |  -ln(1-t)            | |  ln t
   *   Li (x)  =    |    -------- dt    =    |    ------ dt    =   spence(1-x) .
   *     2        | |       t              | |    1 - t
   *               -                        -
   *                0                        1
   *
   *
   *  See also the program cpolylog.c for the complex polylogarithm,
   *  whose definition is extended to x > 1.
   *
   *  References:
   *
   *  Lewin, L., _Polylogarithms and Associated Functions_,
   *  North Holland, 1981.
   *
   *  Lewin, L., ed., _Structural Properties of Polylogarithms_,
   *  American Mathematical Society, 1991.
   *
   *
   * ACCURACY:
   *
   *                      Relative error:
   * arithmetic   domain   n   # trials      peak         rms
   *    IEEE      0, 1     2     50000      6.2e-16     8.0e-17
   *    IEEE      0, 1     3    100000      2.5e-16     6.6e-17
   *    IEEE      0, 1     4     30000      1.7e-16     4.9e-17
   *    IEEE      0, 1     5     30000      5.1e-16     7.8e-17
   *
   */

  /*
    Cephes Math Library Release 2.8:  July, 1999
    Copyright 1999 by Stephen L. Moshier
  */

  /* polylog(4, 1-x) = zeta(4) - x zeta(3) + x^2 A4(x)/B4(x)
     0 <= x <= 0.125
     Theoretical peak absolute error 4.5e-18  */
  static double A4[13] = {
    3.056144922089490701751E-2,
    3.243086484162581557457E-1,
    2.877847281461875922565E-1,
    7.091267785886180663385E-2,
    6.466460072456621248630E-3,
    2.450233019296542883275E-4,
    4.031655364627704957049E-6,
    2.884169163909467997099E-8,
    8.680067002466594858347E-11,
    1.025983405866370985438E-13,
    4.233468313538272640380E-17,
    4.959422035066206902317E-21,
    1.059365867585275714599E-25,
  };
  static double B4[12] = {
    /* 1.000000000000000000000E0, */
    2.821262403600310974875E0,
    1.780221124881327022033E0,
    3.778888211867875721773E-1,
    3.193887040074337940323E-2,
    1.161252418498096498304E-3,
    1.867362374829870620091E-5,
    1.319022779715294371091E-7,
    3.942755256555603046095E-10,
    4.644326968986396928092E-13,
    1.913336021014307074861E-16,
    2.240041814626069927477E-20,
    4.784036597230791011855E-25,
  };

  double polylog (int n, double x) {
    double h, k, p, s, t, u, xc, z;
    int i, j;

    /*  This recurrence provides formulas for n < 2.

        d                 1
        --   Li (x)  =   ---  Li   (x)  .
        dx     n          x     n-1

    */

    if (n == -1)
    {
      p  = 1.0 - x;
      u = x / p;
      s = u * u + u;
      return s;
    }

    if (n == 0)
    {
      s = x / (1.0 - x);
      return s;
    }

    /* Not implemented for n < -1.
       Not defined for x > 1.  Use cpolylog if you need that.  */
    if (x > 1.0 || n < -1)
    {
      report_error("Domain error in polylog");
      return 0.0;
    }

    if (n == 1)
    {
      s = -log (1.0 - x);
      return s;
    }

    /* Argument +1 */
    if (x == 1.0 && n > 1)
    {
      s = zetac ((double) n) + 1.0;
      return s;
    }

    /* Argument -1.
       1-n
       Li (-z)  = - (1 - 2   ) Li (z)
       n                       n
    */
    if (x == -1.0 && n > 1)
    {
      /* Li_n(1) = zeta(n) */
      s = zetac ((double) n) + 1.0;
      s = s * (powi (2.0, 1 - n) - 1.0);
      return s;
    }

    /*  Inversion formula:
     *                                                   [n/2]   n-2r
     *                n                  1     n           -  log    (z)
     *  Li (-z) + (-1)  Li (-1/z)  =  - --- log (z)  +  2  >  ----------- Li  (-1)
     *    n               n              n!                -   (n - 2r)!    2r
     *                                                    r=1
     */
    if (x < -1.0 && n > 1)
    {
      double q, w;
      int r;

      w = log (-x);
      s = 0.0;
      for (r = 1; r <= n / 2; r++)
      {
        j = 2 * r;
        p = polylog (j, -1.0);
        j = n - j;
        if (j == 0)
        {
          s = s + p;
          break;
        }
        q = (double) j;
        q = pow (w, q) * p / fac (j);
        s = s + q;
      }
      s = 2.0 * s;
      q = polylog (n, 1.0 / x);
      if (n & 1)
        q = -q;
      s = s - q;
      s = s - pow (w, (double) n) / fac (n);
      return s;
    }

    if (n == 2)
    {
      if (x < 0.0 || x > 1.0)
        return (spence (1.0 - x));
    }



    /*  The power series converges slowly when x is near 1.  For n = 3, this
        identity helps:

        Li (-x/(1-x)) + Li (1-x) + Li (x)
        3               3          3
        2                               2                 3
        = Li (1) + (pi /6) log(1-x) - (1/2) log(x) log (1-x) + (1/6) log (1-x)
        3
    */

    if (n == 3)
    {
      p = x * x * x;
      if (x > 0.8)
      {
        /* Thanks to Oscar van Vlijmen for detecting an error here.  */
        u = log(x);
        s = u * u * u / 6.0;
        xc = 1.0 - x;
        s = s - 0.5 * u * u * log(xc);
        s = s + PI * PI * u / 6.0;
        s = s - polylog (3, -xc/x);
        s = s - polylog (3, xc);
        s = s + zetac(3.0);
        s = s + 1.0;
        return s;
      }
      /* Power series  */
      t = p / 27.0;
      t = t + .125 * x * x;
      t = t + x;

      s = 0.0;
      k = 4.0;
      do
      {
        p = p * x;
        h = p / (k * k * k);
        s = s + h;
        k += 1.0;
      }
      while (fabs(h/s) > 1.1e-16);
      return (s + t);
    }

    if (n == 4)
    {
      if (x >= 0.875)
      {
        u = 1.0 - x;
        s = polevl(u, A4, 12) / p1evl(u, B4, 12);
        s =  s * u * u - 1.202056903159594285400 * u;
        s +=  1.0823232337111381915160;
        return s;
      }
      goto pseries;
    }


    if (x < 0.75)
      goto pseries;


    /*  This expansion in powers of log(x) is especially useful when
        x is near 1.

        See also the pari gp calculator.

        inf                  j
        -    z(n-j) (log(x))
        polylog(n,x)  =    >   -----------------
        -           j!
        j=0

        where

        z(j) = Riemann zeta function (j), j != 1

        n-1
        -
        z(1) =  -log(-log(x)) +  >  1/k
        -
        k=1
    */

    z = log(x);
    h = -log(-z);
    for (i = 1; i < n; i++)
      h = h + 1.0/i;
    p = 1.0;
    s = zetac((double)n) + 1.0;
    for (j=1; j<=n+1; j++)
    {
      p = p * z / j;
      if (j == n-1)
        s = s + h * p;
      else
        s = s + (zetac((double)(n-j)) + 1.0) * p;
    }
    j = n + 3;
    z = z * z;
    for(;;)
    {
      p = p * z / ((j-1)*j);
      h = (zetac((double)(n-j)) + 1.0);
      h = h * p;
      s = s + h;
      if (fabs(h/s) < MACHEP)
        break;
      j += 2;
    }
    return s;

 pseries:

    p = x * x * x;
    k = 3.0;
    s = 0.0;
    do
    {
      p = p * x;
      k += 1.0;
      h = p / powi(k, n);
      s = s + h;
    }
    while (fabs(h/s) > MACHEP);
    s += x * x * x / powi(3.0,n);
    s += x * x / powi(2.0,n);
    s += x;
    return s;
  }
  }  // namespace Cephes
}  // namespace BOOM

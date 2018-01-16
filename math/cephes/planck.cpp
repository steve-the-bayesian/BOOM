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

  /*                                                    planck.c
   *
   *    Integral of Planck's black body radiation formula
   *
   *
   *
   * SYNOPSIS:
   *
   * double lambda, T, y, plancki();
   *
   * y = plancki( lambda, T );
   *
   *
   *
   * DESCRIPTION:
   *
   *  Evaluates the definite integral, from wavelength 0 to lambda,
   *  of Planck's radiation formula
   *                      -5
   *            c1  lambda
   *     E =  ------------------
   *            c2/(lambda T)
   *           e             - 1
   *
   * Physical constants c1 = 3.7417749e-16 and c2 = 0.01438769 are built in
   * to the function program.  They are scaled to provide a result
   * in watts per square meter.  Argument T represents temperature in degrees
   * Kelvin; lambda is wavelength in meters.
   *
   * The integral is expressed in closed form, in terms of polylogarithms
   * (see polylog.c).
   *
   * The total area under the curve is
   *      (-1/8) (42 zeta(4) - 12 pi^2 zeta(2) + pi^4 ) c1 (T/c2)^4
   *       = (pi^4 / 15)  c1 (T/c2)^4
   *       =  5.6705032e-8 T^4
   * where sigma = 5.6705032e-8 W m^2 K^-4 is the Stefan-Boltzmann constant.
   *
   *
   * ACCURACY:
   *
   * The left tail of the function experiences some relative error
   * amplification in computing the dominant term exp(-c2/(lambda T)).
   * For the right-hand tail see planckc, below.
   *
   *                      Relative error.
   *   The domain refers to lambda T / c2.
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      0.1, 10      50000      7.1e-15     5.4e-16
   *
   */

  /*
    Cephes Math Library Release 2.8:  July, 1999
    Copyright 1999 by Stephen L. Moshier
  */

  /*  NIST value (1999): 2 pi h c^2 = 3.741 7749(22) × 10-16 W m2  */
  const double planck_c1 = 3.7417749e-16;
  /*  NIST value (1999):  h c / k  = 0.014 387 69 m K */
  const double planck_c2 = 0.01438769;
  double planckc(double w, double T);
  double polylog(int n, double x);

  double plancki(double w, double T) {
    double b, h, y, bw;

    b = T / planck_c2;
    bw = b * w;

    if (bw > 0.59375)
    {
      y = b * b;
      h = y * y;
      /* Right tail.  */
      y = planckc (w, T);
      /* pi^4 / 15  */
      y =  6.493939402266829149096 * planck_c1 * h  -  y;
      return y;
    }

    h = exp(-planck_c2/(w*T));
    y =      6. * polylog (4, h)  * bw;
    y = (y + 6. * polylog (3, h)) * bw;
    y = (y + 3. * polylog (2, h)) * bw;
    y = (y          - log1p (-h)) * bw;
    h = w * w;
    h = h * h;
    y = y * (planck_c1 / h);
    return y;
  }

  /*                                                    planckc
   *
   *    Complemented Planck radiation integral
   *
   *
   *
   * SYNOPSIS:
   *
   * double lambda, T, y, planckc();
   *
   * y = planckc( lambda, T );
   *
   *
   *
   * DESCRIPTION:
   *
   *  Integral from w to infinity (area under right hand tail)
   *  of Planck's radiation formula.
   *
   *  The program for large lambda uses an asymptotic series in inverse
   *  powers of the wavelength.
   *
   * ACCURACY:
   *
   *                      Relative error.
   *   The domain refers to lambda T / c2.
   * arithmetic   domain     # trials      peak         rms
   *    IEEE      0.6, 10      50000      1.1e-15     2.2e-16
   *
   */

    double planckc (double w, double T) {
      double b, d, p, u, y;

      b = T / planck_c2;
      d = b*w;
      if (d <= 0.59375)
      {
        y =  6.493939402266829149096 * planck_c1 * b*b*b*b;
        return (y - plancki(w,T));
      }
      u = 1.0/d;
      p = u * u;
      y = -236364091.*p/45733251691757079075225600000.;
      y = (y + 77683./352527500984795136000000.)*p;
      y = (y - 174611./18465726242060697600000.)*p;
      y = (y + 43867./107290978560589824000.)*p;
      y = ((y - 3617./202741834014720000.)*p + 1./1270312243200.)*p;
      y = ((y - 691./19615115520000.)*p + 1./622702080.)*p;
      y = ((((y - 1./13305600.)*p + 1./272160.)*p - 1./5040.)*p + 1./60.)*p;
      y = y - 0.125*u + 1./3.;
      y = y * planck_c1 * b / (w*w*w);
      return y;
    }

  /*                                                    planckd
   *
   *    Planck's black body radiation formula
   *
   *
   *
   * SYNOPSIS:
   *
   * double lambda, T, y, planckd();
   *
   * y = planckd( lambda, T );
   *
   *
   *
   * DESCRIPTION:
   *
   *  Evaluates Planck's radiation formula
   *                      -5
   *            c1  lambda
   *     E =  ------------------
   *            c2/(lambda T)
   *           e             - 1
   *
   */

    double planckd(double w, double T) {
      return (planck_c2 / ((w*w*w*w*w) * (exp(planck_c2/(w*T)) - 1.0)));
    }


  /* Wavelength, w, of maximum radiation at given temperature T.
     c2/wT = constant
     Wein displacement law.
  */
    double planckw(double T) {
      return (planck_c2 / (4.96511423174427630 * T));
    }
  }  // namespace Cephes
}  // namespace BOOM

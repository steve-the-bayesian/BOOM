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

  /*                                                    shichi.c
   *
   *    Hyperbolic sine and cosine integrals
   *
   *
   *
   * SYNOPSIS:
   *
   * double x, Chi, Shi, shichi();
   *
   * shichi( x, &Chi, &Shi );
   *
   *
   * DESCRIPTION:
   *
   * Approximates the integrals
   *
   *                            x
   *                            -
   *                           | |   cosh t - 1
   *   Chi(x) = eul + ln x +   |    -----------  dt,
   *                         | |          t
   *                          -
   *                          0
   *
   *               x
   *               -
   *              | |  sinh t
   *   Shi(x) =   |    ------  dt
   *            | |       t
   *             -
   *             0
   *
   * where eul = 0.57721566490153286061 is Euler's constant.
   * The integrals are evaluated by power series for x < 8
   * and by Chebyshev expansions for x between 8 and 88.
   * For large x, both functions approach exp(x)/2x.
   * Arguments greater than 88 in magnitude return MAXNUM.
   *
   *
   * ACCURACY:
   *
   * Test interval 0 to 88.
   *                      Relative error:
   * arithmetic   function  # trials      peak         rms
   *    DEC          Shi       3000       9.1e-17
   *    IEEE         Shi      30000       6.9e-16     1.6e-16
   *        Absolute error, except relative when |Chi| > 1:
   *    DEC          Chi       2500       9.3e-17
   *    IEEE         Chi      30000       8.4e-16     1.4e-16
   */

  /*
    Cephes Math Library Release 2.8:  June, 2000
    Copyright 1984, 1987, 2000 by Stephen L. Moshier
  */

  /* x exp(-x) shi(x), inverted interval 8 to 18 */
  static double S1[] = {
    1.83889230173399459482E-17,
    -9.55485532279655569575E-17,
    2.04326105980879882648E-16,
    1.09896949074905343022E-15,
    -1.31313534344092599234E-14,
    5.93976226264314278932E-14,
    -3.47197010497749154755E-14,
    -1.40059764613117131000E-12,
    9.49044626224223543299E-12,
    -1.61596181145435454033E-11,
    -1.77899784436430310321E-10,
    1.35455469767246947469E-9,
    -1.03257121792819495123E-9,
    -3.56699611114982536845E-8,
    1.44818877384267342057E-7,
    7.82018215184051295296E-7,
    -5.39919118403805073710E-6,
    -3.12458202168959833422E-5,
    8.90136741950727517826E-5,
    2.02558474743846862168E-3,
    2.96064440855633256972E-2,
    1.11847751047257036625E0
  };

  /* x exp(-x) shi(x), inverted interval 18 to 88 */
  static double S2[] = {
    -1.05311574154850938805E-17,
    2.62446095596355225821E-17,
    8.82090135625368160657E-17,
    -3.38459811878103047136E-16,
    -8.30608026366935789136E-16,
    3.93397875437050071776E-15,
    1.01765565969729044505E-14,
    -4.21128170307640802703E-14,
    -1.60818204519802480035E-13,
    3.34714954175994481761E-13,
    2.72600352129153073807E-12,
    1.66894954752839083608E-12,
    -3.49278141024730899554E-11,
    -1.58580661666482709598E-10,
    -1.79289437183355633342E-10,
    1.76281629144264523277E-9,
    1.69050228879421288846E-8,
    1.25391771228487041649E-7,
    1.16229947068677338732E-6,
    1.61038260117376323993E-5,
    3.49810375601053973070E-4,
    1.28478065259647610779E-2,
    1.03665722588798326712E0
  };

  /* x exp(-x) chin(x), inverted interval 8 to 18 */
  static double C1[] = {
    -8.12435385225864036372E-18,
    2.17586413290339214377E-17,
    5.22624394924072204667E-17,
    -9.48812110591690559363E-16,
    5.35546311647465209166E-15,
    -1.21009970113732918701E-14,
    -6.00865178553447437951E-14,
    7.16339649156028587775E-13,
    -2.93496072607599856104E-12,
    -1.40359438136491256904E-12,
    8.76302288609054966081E-11,
    -4.40092476213282340617E-10,
    -1.87992075640569295479E-10,
    1.31458150989474594064E-8,
    -4.75513930924765465590E-8,
    -2.21775018801848880741E-7,
    1.94635531373272490962E-6,
    4.33505889257316408893E-6,
    -6.13387001076494349496E-5,
    -3.13085477492997465138E-4,
    4.97164789823116062801E-4,
    2.64347496031374526641E-2,
    1.11446150876699213025E0
  };

  /* x exp(-x) chin(x), inverted interval 18 to 88 */
  static double C2[] = {
    8.06913408255155572081E-18,
    -2.08074168180148170312E-17,
    -5.98111329658272336816E-17,
    2.68533951085945765591E-16,
    4.52313941698904694774E-16,
    -3.10734917335299464535E-15,
    -4.42823207332531972288E-15,
    3.49639695410806959872E-14,
    6.63406731718911586609E-14,
    -3.71902448093119218395E-13,
    -1.27135418132338309016E-12,
    2.74851141935315395333E-12,
    2.33781843985453438400E-11,
    2.71436006377612442764E-11,
    -2.56600180000355990529E-10,
    -1.61021375163803438552E-9,
    -4.72543064876271773512E-9,
    -3.00095178028681682282E-9,
    7.79387474390914922337E-8,
    1.06942765566401507066E-6,
    1.59503164802313196374E-5,
    3.49592575153777996871E-4,
    1.28475387530065247392E-2,
    1.03665693917934275131E0
  };

  /* Sine and cosine integrals */

    int shichi(double x, double *si, double *ci ) {
      double k, z, c, s, a;
      short sign;

      if( x < 0.0 )
      {
        sign = -1;
        x = -x;
      }
      else
        sign = 0;


      if( x == 0.0 )
      {
        *si = 0.0;
        *ci = -MAXNUM;
        return( 0 );
      }

      if( x >= 8.0 )
        goto chb;

      z = x * x;

      /*        Direct power series expansion   */

      a = 1.0;
      s = 1.0;
      c = 0.0;
      k = 2.0;

      do
      {
        a *= z/k;
        c += a/k;
        k += 1.0;
        a /= k;
        s += a/k;
        k += 1.0;
      }
      while( fabs(a/s) > MACHEP );

      s *= x;
      goto done;


   chb:

      if( x < 18.0 )
      {
        a = (576.0/x - 52.0)/10.0;
        k = exp(x) / x;
        s = k * chbevl( a, S1, 22 );
        c = k * chbevl( a, C1, 23 );
        goto done;
      }

      if( x <= 88.0 )
      {
        a = (6336.0/x - 212.0)/70.0;
        k = exp(x) / x;
        s = k * chbevl( a, S2, 23 );
        c = k * chbevl( a, C2, 24 );
        goto done;
      }
      else
      {
        if( sign )
          *si = -MAXNUM;
        else
          *si = MAXNUM;
        *ci = MAXNUM;
        return(0);
      }
   done:
      if( sign )
        s = -s;

      *si = s;

      *ci = eulers_constant + log(x) + c;
      return(0);
    }
  }  // namespace Cephes
}  // namespace BOOM

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
  /*                                                    ei.c
   *
   *    Exponential integral
   *
   *
   * SYNOPSIS:
   *
   * double x, y, ei();
   *
   * y = ei( x );
   *
   *
   *
   * DESCRIPTION:
   *
   *               x
   *                -     t
   *               | |   e
   *    Ei(x) =   -|-   ---  dt .
   *             | |     t
   *              -
   *             -inf
   *
   * Not defined for x <= 0.
   * See also expn.c.
   *
   *
   *
   * ACCURACY:
   *
   *                      Relative error:
   * arithmetic   domain     # trials      peak         rms
   *    IEEE       0,100       50000      8.6e-16     1.3e-16
   *
   */

  /*
    Cephes Math Library Release 2.8:  May, 1999
    Copyright 1999 by Stephen L. Moshier
  */

  /* 0 < x <= 2
     Ei(x) - eulers_constant - ln(x) = x A(x)/B(x)
     Theoretical peak relative error 9.73e-18  */
  const double A[] = {
      -5.350447357812542947283E0,
          2.185049168816613393830E2,
          -4.176572384826693777058E3,
          5.541176756393557601232E4,
          -3.313381331178144034309E5,
          1.592627163384945414220E6,
          };
  const double B[] = {
      /*  1.000000000000000000000E0, */
      -5.250547959112862969197E1,
          1.259616186786790571525E3,
          -1.756549581973534652631E4,
          1.493062117002725991967E5,
          -7.294949239640527645655E5,
          1.592627163384945429726E6,
          };

  /* 8 <= x <= 20
     x exp(-x) Ei(x) - 1 = 1/x R(1/x)
     Theoretical peak absolute error = 1.07e-17  */
  const double A2[] = {
      -2.106934601691916512584E0,
          1.732733869664688041885E0,
          -2.423619178935841904839E-1,
          2.322724180937565842585E-2,
          2.372880440493179832059E-4,
          -8.343219561192552752335E-5,
          1.363408795605250394881E-5,
          -3.655412321999253963714E-7,
          1.464941733975961318456E-8,
          6.176407863710360207074E-10,
          };

  const double B2[] = {
      /* 1.000000000000000000000E0, */
      -2.298062239901678075778E-1,
          1.105077041474037862347E-1,
          -1.566542966630792353556E-2,
          2.761106850817352773874E-3,
          -2.089148012284048449115E-4,
          1.708528938807675304186E-5,
          -4.459311796356686423199E-7,
          1.394634930353847498145E-8,
          6.150865933977338354138E-10,
          };

  /* x > 20
     x exp(-x) Ei(x) - 1  =  1/x A3(1/x)/B3(1/x)
     Theoretical absolute error = 6.15e-17  */

  const double A3[] = {
      -7.657847078286127362028E-1,
          6.886192415566705051750E-1,
          -2.132598113545206124553E-1,
          3.346107552384193813594E-2,
          -3.076541477344756050249E-3,
          1.747119316454907477380E-4,
          -6.103711682274170530369E-6,
          1.218032765428652199087E-7,
          -1.086076102793290233007E-9,
          };

  const double B3[] = {
      /* 1.000000000000000000000E0, */
      -1.888802868662308731041E0,
          1.066691687211408896850E0,
          -2.751915982306380647738E-1,
          3.930852688233823569726E-2,
          -3.414684558602365085394E-3,
          1.866844370703555398195E-4,
          -6.345146083130515357861E-6,
          1.239754287483206878024E-7,
          -1.086076102793126632978E-9,
          };

  /* 16 <= x <= 32
     x exp(-x) Ei(x) - 1  =  1/x A4(1/x) / B4(1/x)
     Theoretical absolute error = 1.22e-17  */
  const double A4[] = {
      -2.458119367674020323359E-1,
          -1.483382253322077687183E-1,
          7.248291795735551591813E-2,
          -1.348315687380940523823E-2,
          1.342775069788636972294E-3,
          -7.942465637159712264564E-5,
          2.644179518984235952241E-6,
          -4.239473659313765177195E-8,
          };

  const double B4[] = {
      /* 1.000000000000000000000E0, */
      -1.044225908443871106315E-1,
          -2.676453128101402655055E-1,
          9.695000254621984627876E-2,
          -1.601745692712991078208E-2,
          1.496414899205908021882E-3,
          -8.462452563778485013756E-5,
          2.728938403476726394024E-6,
          -4.239462431819542051337E-8,
          };

  /* 4 <= x <= 8
     x exp(-x) Ei(x) - 1  =  1/x A5(1/x) / B5(1/x)
     Theoretical absolute error = 2.20e-17  */
  const double A5[] = {
      -1.373215375871208729803E0,
          -7.084559133740838761406E-1,
          1.580806855547941010501E0,
          -2.601500427425622944234E-1,
          2.994674694113713763365E-2,
          -1.038086040188744005513E-3,
          4.371064420753005429514E-5,
          2.141783679522602903795E-6,
          };

  const double B5[] = {
      /* 1.000000000000000000000E0, */
      8.585231423622028380768E-1,
          4.483285822873995129957E-1,
          7.687932158124475434091E-2,
          2.449868241021887685904E-2,
          8.832165941927796567926E-4,
          4.590952299511353531215E-4,
          -4.729848351866523044863E-6,
          2.665195537390710170105E-6,
          };

  /* 2 <= x <= 4
     x exp(-x) Ei(x) - 1  =  1/x A6(1/x) / B6(1/x)
     Theoretical absolute error = 4.89e-17  */
  const double A6[] = {
      1.981808503259689673238E-2,
          -1.271645625984917501326E0,
          -2.088160335681228318920E0,
          2.755544509187936721172E0,
          -4.409507048701600257171E-1,
          4.665623805935891391017E-2,
          -1.545042679673485262580E-3,
          7.059980605299617478514E-5,
          };

  const double B6[] = {
      /* 1.000000000000000000000E0, */
      1.476498670914921440652E0,
          5.629177174822436244827E-1,
          1.699017897879307263248E-1,
          2.291647179034212017463E-2,
          4.450150439728752875043E-3,
          1.727439612206521482874E-4,
          3.953167195549672482304E-5,
          };

  /* 32 <= x <= 64
     x exp(-x) Ei(x) - 1  =  1/x A7(1/x) / B7(1/x)
     Theoretical absolute error = 7.71e-18  */
  const double A7[] = {
      1.212561118105456670844E-1,
          -5.823133179043894485122E-1,
          2.348887314557016779211E-1,
          -3.040034318113248237280E-2,
          1.510082146865190661777E-3,
          -2.523137095499571377122E-5,
          };

  const double B7[] = {
      /* 1.000000000000000000000E0, */
      -1.002252150365854016662E0,
          2.928709694872224144953E-1,
          -3.337004338674007801307E-2,
          1.560544881127388842819E-3,
          -2.523137093603234562648E-5,
          };

  double ei (double x){
    double f, w;

    if (x <= 0.0)
    {
      report_error("Domain error in ei.  x < 0.");
      return 0.0;
    }
    else if (x < 2.0)
    {
      /* Power series.
         inf    n
         -    x
         Ei(x) = eulers_constant + ln x  +   >   ----
         -   n n!
         n=1
      */
      f = polevl(x,A,5) / p1evl(x,B,6);
      /*      f = polevl(x,A,6) / p1evl(x,B,7); */
      /*      f = polevl(x,A,8) / p1evl(x,B,9); */
      return (eulers_constant + log(x) + x * f);
    }
    else if (x < 4.0)
    {
      /* Asymptotic expansion.
         1       2       6
         x exp(-x) Ei(x) =  1 + ---  +  ---  +  ---- + ...
         x        2       3
         x       x
      */
      w = 1.0/x;
      f = polevl(w,A6,7) / p1evl(w,B6,7);
      return (exp(x) * w * (1.0 + w * f));
    }
    else if (x < 8.0)
    {
      w = 1.0/x;
      f = polevl(w,A5,7) / p1evl(w,B5,8);
      return (exp(x) * w * (1.0 + w * f));
    }
    else if (x < 16.0)
    {
      w = 1.0/x;
      f = polevl(w,A2,9) / p1evl(w,B2,9);
      return (exp(x) * w * (1.0 + w * f));
    }
    else if (x < 32.0)
    {
      w = 1.0/x;
      f = polevl(w,A4,7) / p1evl(w,B4,8);
      return (exp(x) * w * (1.0 + w * f));
    }
    else if (x < 64.0)
    {
      w = 1.0/x;
      f = polevl(w,A7,5) / p1evl(w,B7,5);
      return (exp(x) * w * (1.0 + w * f));
    }
    else
    {
      w = 1.0/x;
      f = polevl(w,A3,8) / p1evl(w,B3,9);
      return (exp(x) * w * (1.0 + w * f));
    }
  }
  }  // namespace Cephes
}  // namespace BOOM

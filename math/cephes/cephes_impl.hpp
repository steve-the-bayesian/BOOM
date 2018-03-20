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

#ifndef BOOM_MATH_CEPHES_HPP_
#define BOOM_MATH_CEPHES_HPP_

#include <limits>
#include "cpputil/report_error.hpp"
#include <cmath>

namespace BOOM {
  namespace Cephes {
    const double eulers_constant = 0.57721566490153286060;
    const double MACHEP = std::numeric_limits<double>::epsilon();
    const double MAXNUM = std::numeric_limits<double>::max();
    const double MAXLOG = log(MAXNUM);
    const double PI     =  3.14159265358979323846;       // pi
    const double PIO2   =  1.57079632679489661923;       // pi/2
    const double LOGE2  =  6.93147180559945309417E-1;    // log(2)

    double chbevl(double x, double *array, int n);
    double fac(int i);
    double p1evl(double x, const double *coef, int N);
    double polevl(double x, const double *coef, int N);
    double polylog(int n, double x);
    double powi(double x, int n);
    double spence(double x);
    double zeta(double x, double q);
    double zetac(double x);
  }
}

#endif //  BOOM_MATH_CEPHES_HPP_

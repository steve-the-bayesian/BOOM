// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#include "numopt/ScalarNewtonMax.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {
  double scalar_newton_max(const d2ScalarTargetFun &f, double &x, double &g,
                           double &h) {
    double y = f(x, g, h);
    double oldy = y;
    double eps = 1e-5;  // TODO:  remove magic numbers
    double dy = eps + 1;
    while (dy > eps) {
      double step = g / h;
      if (h > 0) step = -1 * step;
      double oldx = x;
      x -= step;
      y = f(x, g, h);
      dy = y - oldy;
      while (dy < 0) {
        step /= 2.0;
        x = oldx - step;
        if (fabs(step) < eps) {
          report_error("too small a step size in scalar_newton_max");
        }
        y = f(x, g, h);
        dy = y - oldy;
      }
      oldy = y;
    }
    return y;
  }
}  // namespace BOOM

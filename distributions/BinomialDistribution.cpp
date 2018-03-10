// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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

#include "distributions/BinomialDistribution.hpp"

namespace BOOM {

  binomial_distribution::binomial_distribution(uint nin, double pp) : n(nin) {
    setup(pp);
  }
  //----------------------------------------------------------------------
  void binomial_distribution::setup(double pp) {
    np = n * p;
    p = std::min(pp, 1. - pp);
    q = 1. - p;
    np = n * p;
    r = p / q;
    g = r * (n + 1);
    psave = pp;

    if (np < 30) {
      qn = pow(q, (double)n);
      return;
    } else {
      ffm = np + p;
      m = static_cast<int>(ffm);
      fm = m;
      npq = np * q;
      p1 = (int)(2.195 * sqrt(npq) - 4.6 * q) + 0.5;
      xm = fm + 0.5;
      xl = xm - p1;
      xr = xm + p1;
      c = 0.134 + 20.5 / (15.3 + fm);
      al = (ffm - xl) / (ffm - xl * p);
      xll = al * (1.0 + 0.5 * al);
      al = (xr - ffm) / (xr * q);
      xlr = al * (1.0 + 0.5 * al);
      p2 = p1 * (1.0 + c + c);
      p3 = p2 + c / xll;
      p4 = p3 + c / xlr;
    }
  }
  //----------------------------------------------------------------------
  uint binomial_distribution::operator()(RNG &rng) {
    if (np < 30) return draw_np_small(rng);
    while (true) {
      u = rng() * p4;
      v = rng();
      /* triangular region */
      if (u <= p1) {
        ix = static_cast<int>(xm - p1 * v + u);
        return finis();
      }
      /* parallelogram region */
      if (u <= p2) {
        x = xl + (u - p1) / c;
        v = v * c + 1.0 - fabs(xm - x) / p1;
        if (v > 1.0 || v <= 0.) continue;
        ix = static_cast<int>(x);
      } else {
        if (u > p3) { /* right tail */
          ix = static_cast<int>(xr - log(v) / xlr);
          if (static_cast<unsigned>(ix) > n) continue;
          v = v * (u - p3) * xlr;
        } else { /* left tail */
          ix = static_cast<int>(xl + log(v) / xll);
          if (ix < 0) continue;
          v = v * (u - p2) * xll;
        }
      }
      /* determine appropriate way to perform accept/reject test */
      k = abs(ix - m);
      if (k <= 20 || k >= npq / 2 - 1) {
        /* explicit evaluation */
        f = 1.0;
        if (m < ix) {
          for (i = m + 1; i <= ix; i++) f *= (g / i - r);
        } else if (m != ix) {
          for (i = ix + 1; i <= m; i++) f /= (g / i - r);
        }
        if (v <= f) return finis();
      } else {
        /* squeezing using upper and lower bounds on log(f(x)) */
        amaxp =
            (k / npq) * ((k * (k / 3. + 0.625) + 0.1666666666666) / npq + 0.5);
        ynorm = -1.0 * k * k / (2.0 * npq);
        alv = log(v);
        if (alv < ynorm - amaxp) return finis();
        if (alv <= ynorm + amaxp) {
          /* stirling's formula to machine accuracy */
          /* for the final acceptance/rejection test */
          x1 = ix + 1;
          f1 = fm + 1.0;
          z = n + 1 - fm;
          w = n - ix + 1.0;
          z2 = z * z;
          x2 = x1 * x1;
          f2 = f1 * f1;
          w2 = w * w;
          if (alv <=
              xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) +
                  (ix - m) * log(w * p / (x1 * q)) +
                  (13860.0 -
                   (462.0 - (132.0 - (99.0 - 140.0 / f2) / f2) / f2) / f2) /
                      f1 / 166320.0 +
                  (13860.0 -
                   (462.0 - (132.0 - (99.0 - 140.0 / z2) / z2) / z2) / z2) /
                      z / 166320.0 +
                  (13860.0 -
                   (462.0 - (132.0 - (99.0 - 140.0 / x2) / x2) / x2) / x2) /
                      x1 / 166320.0 +
                  (13860.0 -
                   (462.0 - (132.0 - (99.0 - 140.0 / w2) / w2) / w2) / w2) /
                      w / 166320.)
            return finis();
        }
      }
    }
    return draw_np_small(rng);
  }

  //----------------------------------------------------------------------
  uint binomial_distribution::finis() {
    if (psave > 0.5) ix = n - ix;
    return static_cast<unsigned>(ix);
  }
  //----------------------------------------------------------------------
  uint binomial_distribution::draw_np_small(RNG &rng) {
    while (true) {
      ix = 0;
      f = qn;
      u = rng();
      while (true) {
        if (u < f) return finis();
        if (ix > 110) break;
        u -= f;
        ix++;
        f *= (g / ix - r);
      }
    }
    return finis();
  }

}  // namespace BOOM

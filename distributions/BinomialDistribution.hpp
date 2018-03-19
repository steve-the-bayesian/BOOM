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

#ifndef BOOM_BINOMIAL_DISTRIBUTION_HPP
#define BOOM_BINOMIAL_DISTRIBUTION_HPP

#include "distributions/rng.hpp"
#include "uint.hpp"

namespace BOOM {

  class binomial_distribution {
    // different case models tr1::binomial_distribution
   public:
    explicit binomial_distribution(uint n = 1, double p = 0.5);
    uint operator()(RNG &);

   private:
    void setup(double pp);
    uint finis();
    uint draw_np_small(RNG &);

    double c, fm, npq, p1, p2, p3, p4, qn;
    double xl, xll, xlr, xm, xr;
    double psave;
    int m;

    double f, f1, f2, u, v, w, w2, x, x1, x2, z, z2;
    double p, q, np, g, r, al, alv, amaxp, ffm, ynorm;
    int i, k, ix;
    unsigned n;
  };

}  // namespace BOOM
#endif  // BOOM_BINOMIAL_DISTRIBUTION_HPP

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
#include <cmath>
#include <sstream>
#include "LinAlg/CorrelationMatrix.hpp"
#include "Samplers/UnivariateSliceSampler.hpp"
#include "TargetFun/TargetFun.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {
  typedef CorrelationMatrix CM;

  class Rdet {
   public:
    Rdet(const CM &R, int I, int J) : R_(R), i(I), j(J) {}
    double operator()(double r) {
      R_(i, j) = r;
      R_(j, i) = r;
      double ans = R_.det();
      if (isnan(ans)) {
        std::ostringstream err;
        err << "Illegal determinant in random_cor:  R = " << std::endl
            << R_ << std::endl;
        report_error(err.str());
      }
      return ans;
    }

   private:
    Matrix R_;
    int i, j;
  };

  // This function is supposed to draw a random correlation matrix
  // from the uniform distribution on the space of all correlation
  // matrices.  It is broken
  CM random_cor_mt(RNG &rng, uint n) {
    CM R(n);
    for (int k = 0; k < 1; ++k) {
      for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
          Rdet f(R, i, j);
          double f1 = f(1);
          double fn = f(-1);
          double f0 = f(0);
          double a = .5 * (f1 + fn - 2 * f0);
          double b = .5 * (f1 - fn);
          double c = f0;

          double d2 = b * b - 4 * a * c;
          if (d2 < 0) {
            R(i, j) = 0;
            R(j, i) = 0;
            continue;
          }
          double d = std::sqrt(d2);
          double lo = (-b - d) / (2 * a);
          double hi = (-b + d) / (2 * a);
          if (a < 0) std::swap(lo, hi);
          double r = runif_mt(rng, lo, hi);
          R(i, j) = r;
          R(j, i) = r;
        }
      }
    }
    return R;
  }
}  // namespace BOOM

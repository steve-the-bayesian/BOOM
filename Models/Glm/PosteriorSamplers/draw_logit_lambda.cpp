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

#include "Models/Glm/PosteriorSamplers/draw_logit_lambda.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"
#include "distributions/inverse_gaussian.hpp"

namespace BOOM {
  namespace Logit {
    double draw_lambda_mt(RNG &rng, double r) {
      r = fabs(r);
      //       //    if (r<.0001) r=.0001;
      //       double y = rnorm_mt(rng, 0, 1);
      //       y *= y;
      //       y = 1 + (y - sqrt(y * (4*r + y) ) )/(2*r);

      //       double u = runif_mt(rng, 0,1);
      //       double inv = (u < 1/(1+y));
      //       double lam = inv ? r/y : r*y;

      //       // now lam is GIG(.5, 1, r^2)

      double lam = 0;
      if (r < 1e-4) {
        lam = rgamma_mt(rng, .5, .5);
      } else {
        double ilam = rig_mt(rng, 1.0 / r, 1.0);
        lam = 1.0 / ilam;
      }

      bool right = lam > 1.33333333;
      double u = runif_mt(rng, 0, 1);
      bool okay = right ? check_right(u, lam) : check_left(u, lam);
      if (okay) {
        return lam;
      }

      // If we made it here it means the draw was rejected... try again
      return draw_lambda_mt(rng, r);
    }
    //------------------------------------------------------------
    bool check_right(double u, double lam) {
      double z = 1;
      double x = exp(-.5 * lam);
      uint j = 0;

      while (1) {
        ++j;
        double jp1sq = square(j + 1);
        z -= jp1sq * pow(x, jp1sq - 1);
        if (z > u) {
          return true;
        }

        ++j;
        jp1sq = square(j + 1);
        z += jp1sq * pow(x, jp1sq - 1);
        if (z < u) {
          return false;
        }
      }
    }
    //------------------------------------------------------------
    bool check_left(double u, double lam) {
      const double pi_square = 9.86960440108936;
      const double half_pi_square = 4.93480220054468;
      const double half_log_2 = 0.346573590279973;
      const double two_point_five_log_pi = 2.8618247146235;
      double h = half_log_2 + two_point_five_log_pi - 2.5 * log(lam) -
                 half_pi_square / lam + .5 * lam;
      double logu = log(u);

      double z = 1;
      double x = exp(-half_pi_square / lam);
      double k = lam / pi_square;
      uint j = 0;
      while (1) {
        ++j;
        z -= k * pow(x, j * j - 1);
        //      if (z>0 && h + log(z) > logu) return true;
        if (h + log(z) > logu) {
          return true;
        }

        ++j;
        double jp1sq = square(j + 1);
        z += jp1sq * pow(x, jp1sq - 1);
        //      if (z>0 && (h + log(z) < logu) ) return false;
        if (h + log(z) < logu) {
          return false;
        }
      }
    }
  }  // namespace Logit
}  // namespace BOOM

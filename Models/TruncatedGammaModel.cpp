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
#include "Models/TruncatedGammaModel.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  TruncatedGammaModel::TruncatedGammaModel(double a, double b, double lower,
                                           double upper)
      : GammaModel(a, b),
        lower_truncation_point_(lower),
        upper_truncation_point_(upper),
        plo_(pgamma(lower_truncation_point_, a, b, false, false)),
        phi_(pgamma(upper_truncation_point_, a, b, false, false)),
        lognc_(log(phi_ - plo_)) {}

  double TruncatedGammaModel::logp(double x) const {
    if (x < lower_truncation_point_ || x > upper_truncation_point_) {
      return BOOM::negative_infinity();
    } else {
      return dgamma(x, alpha(), beta(), true) - lognc_;
    }
  }

  double TruncatedGammaModel::dlogp(double x, double &derivative) const {
    if (x < lower_truncation_point_) {
      derivative = infinity();
      return negative_infinity();
    } else if (x > upper_truncation_point_) {
      derivative = negative_infinity();
      return negative_infinity();
    } else {
      return GammaModel::dlogp(x, derivative) - lognc_;
    }
  }

  double TruncatedGammaModel::sim(RNG &rng) const {
    static const double threshold = log(.1);
    if (lognc_ > threshold) {
      double ans = negative_infinity();
      do {
        ans = GammaModel::sim(rng);
      } while (ans < lower_truncation_point_ || ans > upper_truncation_point_);
      return ans;
    } else {
      double u = runif_mt(rng, plo_, phi_);
      return qgamma(u, alpha(), beta());
    }
  }

}  // namespace BOOM

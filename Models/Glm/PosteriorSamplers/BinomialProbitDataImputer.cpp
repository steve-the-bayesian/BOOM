// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2016 Steven L. Scott

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

#include "Models/Glm/PosteriorSamplers/BinomialProbitDataImputer.hpp"
#include <cstdint>
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  BinomialProbitDataImputer::BinomialProbitDataImputer(int clt_threshold)
      : clt_threshold_(clt_threshold) {}

  double BinomialProbitDataImputer::impute(RNG &rng, double number_of_trials,
                                           double number_of_successes,
                                           double eta) const {
    int64_t n = lround(number_of_trials);
    int64_t y = lround(number_of_successes);
    if (y < 0 || n < 0) {
      report_error(
          "Negative values not allowed in "
          "BinomialProbitDataImputer::impute().");
    }
    if (y > n) {
      report_error(
          "Success count exceeds trial count in "
          "BinomialProbitDataImputer::impute.");
    }
    double mean, variance;

    double ans = 0;
    if (y > clt_threshold_) {
      trun_norm_moments(eta, 1, 0, true, &mean, &variance);
      // If we draw y deviates from the same truncated normal and add
      // them up we'll have a normal with mean (y * mean) and variance
      // (y * variance).
      ans += rnorm_mt(rng, y * mean, sqrt(y * variance));
    } else {
      for (int i = 0; i < y; ++i) {
        // TODO: If y is large-ish but not quite
        // clt_threshold_ then we might waste some time here
        // constantly rebuilding the same TnSampler object.
        ans += rtrun_norm_mt(rng, eta, 1, 0, true);
      }
    }

    if (n - y > clt_threshold_) {
      trun_norm_moments(eta, 1, 0, false, &mean, &variance);
      ans += rnorm_mt(rng, (n - y) * mean, sqrt((n - y) * variance));
    } else {
      for (int i = 0; i < n - y; ++i) {
        ans += rtrun_norm_mt(rng, eta, 1, 0, false);
      }
    }
    return ans;
  }
}  // namespace BOOM

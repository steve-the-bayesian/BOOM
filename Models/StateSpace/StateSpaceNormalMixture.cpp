// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#include "Models/StateSpace/StateSpaceNormalMixture.hpp"

namespace BOOM {

  StateSpaceNormalMixture::StateSpaceNormalMixture(bool has_regression)
      : has_regression_(has_regression) {}

  Vector StateSpaceNormalMixture::regression_contribution() const {
    if (!has_regression_) {
      return Vector();
    }
    Vector ans(time_dimension());
    for (int time = 0; time < ans.size(); ++time) {
      int nobs = total_sample_size(time);
      double total = 0;
      for (int obs = 0; obs < nobs; ++obs) {
        total += observation_model()->predict(data(time, obs).x());
      }
      ans[time] = nobs > 0 ? total / nobs : 0;
    }
    return ans;
  }

}  // namespace BOOM

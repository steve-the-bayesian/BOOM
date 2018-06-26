#ifndef BOOM_STATE_SPACE_UTILS_HPP_
#define BOOM_STATE_SPACE_UTILS_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

namespace BOOM {

  inline Vector SimulateLocalLevel(RNG &rng, int length, double initial_value,
                            double sd) {
    Vector ans(length);
    ans[0] = initial_value;
    for (int i = 1; i < length; ++i) {
      ans[i] = rnorm_mt(rng, ans[i - 1], sd);
    }
    return ans;
  }

  inline Vector SimulateLocalLinearTrend(
      RNG &rng, int length, double initial_level, double initial_slope,
      double level_sd, double slope_sd) {
    Vector state = {initial_level, initial_slope};
    Vector ans(length);
    Matrix transition = rbind(Vector{1, 1}, Vector{0, 1});
    for (int i = 0; i < length; ++i) {
      ans[i] = state[0];
      state = transition * state;
      state[0] += rnorm_mt(rng, 0, level_sd);
      state[1] += rnorm_mt(rng, 0, slope_sd);
    }
    return ans;
  }
  
  inline Matrix SeasonalStateMatrix(int num_seasons) {
    Matrix ans(num_seasons - 1, num_seasons - 1, 0.0);
    ans.row(0) = -1;
    for (int i = 1; i < ans.nrow(); ++i) {
      ans(i, i-1) = 1.0;
    }
    return ans;
  }
  
  inline Vector SimulateSeasonal(RNG &rng, int length,
                                 const Vector &initial_pattern, double sd) {
    int num_seasons = initial_pattern.size() + 1;
    Matrix transition = SeasonalStateMatrix(num_seasons);
    Vector ans(length);
    Vector state = initial_pattern;
    for (int i = 0; i < length; ++i) {
      state = transition * state;
      state[0] += rnorm_mt(rng, 0, sd);
      ans[i] = state[0];
    }
    return ans;
  }

}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_UTILS_HPP_


/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "stats/acf.hpp"

namespace BOOM {

  Vector acf(const ConstVectorView &x, int num_lags, bool correlation) {
    Vector ans(num_lags + 1);
    int n = x.size();
    for(int lag = 0; lag <= num_lags; ++lag) {
      double sum = 0.0;
      int nu = 0;
      for(int i = 0; i < n-lag; ++i) {
        ++nu;
        sum += x[i + lag] * x[i];
      }
      ans[lag] = (nu > 0) ? sum/(nu + lag) : std::numeric_limits<double>::quiet_NaN();
    }
    if(correlation) {
      if(n == 1) {
        ans[0] = 1.0;
      } else {
        double se = 0;
        se = sqrt(ans[0]);
        for(int lag = 0; lag <= num_lags; lag++) { // ensure correlations remain in  [-1,1] :
          double a = ans[lag] / se;
          ans[lag] = (a > 1.) ? 1. : ((a < -1.) ? -1. : a);
        }
      }
    }
    return ans;
  }

} // namespace BOOM

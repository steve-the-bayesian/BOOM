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

#include "stats/quantile.hpp"
#include "cpputil/report_error.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"

namespace BOOM {

  double quantile(const ConstVectorView &data, double target_quantile) {
    Vector sorted = sort(data);
    return sorted_vector_quantile(sorted, target_quantile);
  }

  Vector quantile(const ConstVectorView &data, const Vector &target_quantiles) {
    Vector sorted = sort(data);
    Vector ans(target_quantiles.size());
    for (int i = 0; i < target_quantiles.size(); ++i) {
      ans[i] = sorted_vector_quantile(sorted, target_quantiles[i]);
    }
    return ans;
  }

  Vector quantile(const Matrix &draws, double target_quantile) {
    Vector ans(draws.ncol());
    for (int i = 0; i < draws.ncol(); ++i) {
      ans[i] = quantile(draws.col(i), target_quantile);
    }
    return ans;
  }

  Matrix quantile(const Matrix &data, const Vector &target_quantiles) {
    Matrix ans(target_quantiles.size(), data.ncol());
    for (int i = 0; i < data.ncol(); ++i){
      ans.col(i) = quantile(data.col(i), target_quantiles);
    }
    return ans;
  }
}

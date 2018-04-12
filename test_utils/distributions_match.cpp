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

#include "test_utils/test_utils.hpp"
#include "stats/ECDF.hpp"
#include "stats/ks_critical_value.hpp"

namespace BOOM {
  bool DistributionsMatch(
      const Vector &data,
      const std::function<double(double)> &cdf,
      double significance) {

    ECDF ecdf(data);
    const Vector &sorted_data(ecdf.sorted_data());
    double maxdiff = negative_infinity();
    for (int64_t i = 0; i < sorted_data.size(); ++i) {
      double delta = fabs(ecdf(sorted_data[i]) - cdf(sorted_data[i]));
      if (delta > maxdiff) {
        maxdiff = delta;
      }
    }
    return maxdiff <= ks_critical_value(sorted_data.size(), significance);
  }

}  // namespace BOOM


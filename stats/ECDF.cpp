/*
  Copyright (C) 2007 Steven L. Scott

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

#include "stats/ECDF.hpp"
#include "cpputil/report_error.hpp"
#include <algorithm>

namespace BOOM{

  ECDF::ECDF(const std::vector<double> &unsorted)
      : sorted_(unsorted)
  {
    if (unsorted.empty()) {
      report_error("ECDF cannot be built from empty vector.");
    }
    std::sort(sorted_.begin(), sorted_.end());
  }

  double ECDF::fplus(double x)const{
    double ans = std::upper_bound(sorted_.begin(), sorted_.end(), x)
        - sorted_.begin();
    return ans / sorted_.size();
  }

  double ECDF::fminus(double x)const{
    double ans = std::lower_bound(sorted_.begin(), sorted_.end(), x)
        - sorted_.begin();
    return ans / sorted_.size();
  }

}  // namespace BOOM

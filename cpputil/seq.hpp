// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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
#include <vector>
#include "cpputil/report_error.hpp"

namespace BOOM {
  // A regular sequence of data.
  // Args:
  //   from: The value of the first element in the sequence.
  //   to: The targeted value of the final element in the sequence.  If 'to' is
  //     not an integral number of steps from 'from' then the sequence stops
  //     with the final element before 'to' is reached.
  //   by: The increment between consecutinv sequence elements.  If 'to' is than
  //     'from' then 'by' should be negative.  It is an error to call this
  //     function with 'by' == 0 unless 'to' == 'from'.
  //
  // Returns:
  //   A vector of the requested type, containing the sequence values.
  template <class NUMERIC>
  std::vector<NUMERIC> seq(const NUMERIC &from, const NUMERIC &to,
                           const NUMERIC &by = 1) {
    std::vector<NUMERIC> ans(1, from);
    if (from == to) {
      return ans;
    }
    int sign = to > from ? 1 : -1;
    if ((sign > 0 && by < 0) || (sign < 0 && by > 0) || by == 0) {
      std::ostringstream err;
      err << "Illegal combination of arguments.  You can't get from " << from
          << " to " << to << " by adding increments of " << by << "."
          << std::endl;
      report_error(err.str());
    }
    size_t space_needed =
        static_cast<size_t>(1u + floor(fabs(double(to - from) / by)));
    ans.reserve(space_needed);
    while (true) {
      NUMERIC tmp = ans.back() + by;
      if ((sign == 1 && tmp > to) || (sign == -1 && tmp < to)) {
        return ans;
      } else {
        ans.push_back(tmp);
      }
    }
    return ans;
  }

}  // namespace BOOM

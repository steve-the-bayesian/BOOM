/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "stats/kl_divergence.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  double kl_divergence(const Vector &p1, const Vector &p2) {
    if (p1.size() != p2.size()) {
      report_error("p1 and p2 must be the same size.");
    }
    double ans = 0;
    for (size_t i = 0; i < p1.size(); ++i) {
      ans += p1[i] * log(p1[i] / p2[i]);
    }
    return ans;
  }
  
}

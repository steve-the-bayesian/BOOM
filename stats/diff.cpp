// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#include "stats/diff.hpp"

namespace BOOM {

  template <class V>
  Vector diff_impl(const V &v, bool leading_zero) {
    int n = v.size();
    if (n == 0) return v;
    Vector ans(leading_zero ? n : n - 1);
    int pos = 0;
    if (leading_zero) {
      ans[0] = 0;
      ++pos;
    }
    for (int i = 1; i < n; ++i) {
      ans[pos++] = v[i] - v[i - 1];
    }
    return ans;
  }

  Vector diff(const Vector &v, bool leading_zero) {
    return diff_impl(v, leading_zero);
  }
  Vector diff(const VectorView &v, bool leading_zero) {
    return diff_impl(v, leading_zero);
  }
  Vector diff(const ConstVectorView &v, bool leading_zero) {
    return diff_impl(v, leading_zero);
  }

}  // namespace BOOM

// Copyright 2018 Google LLC. All Rights Reserved.
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
#include "cpputil/rep.hpp"

namespace BOOM {

  Vector rep(const Vector &x, uint n) {
    uint m = x.size();
    Vector ans(m * n);
    Vector::const_iterator b = x.begin();
    Vector::const_iterator e = x.end();
    Vector::iterator out = ans.begin();
    for (uint i = 0; i < n; ++i) {
      std::copy(b, e, out);
      out += m;
    }
    return ans;
  }
}  // namespace BOOM

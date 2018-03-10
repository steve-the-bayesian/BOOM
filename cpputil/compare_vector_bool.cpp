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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#include "cpputil/compare_vector_bool.hpp"

namespace BOOM {
  typedef std::vector<bool> VB;
  typedef unsigned int uint;

  bool operator==(const VB &x, const VB &y) {
    if (x.size() != y.size()) return false;
    for (uint i = 0; i < x.size(); ++i)
      if (x[i] != y[i]) return false;
    return true;
  }

  bool less(const VB &x, const VB &y) { return less(x, y, false); }

  bool less(const VB &x, const VB &y, bool or_equal) {  // true if x < y
    bool y_is_longer(y.size() > x.size());
    const VB &longer(y_is_longer ? y : x);   // if tied, longer=x
    const VB &shorter(y_is_longer ? x : y);  // if tied, shorter=y
    for (uint i = shorter.size(); i < longer.size(); ++i) {
      if (longer[i]) {                      // longer > shorter
        return y_is_longer ? true : false;  // y_is_longer
      }
    }
    for (uint i = shorter.size() - 1; /* */; --i) {
      if (x[i] != y[i]) return y[i];  // y[i]==1  -> y>x
      if (i == 0) break;
    }
    // made it through entire check with no conclusion... must be equal
    return or_equal;  // i.e. if x<=y is desired answer is true.  if
                      // x<y is desired must be false
  }

}  // namespace BOOM

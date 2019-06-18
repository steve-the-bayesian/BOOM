// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_CPPUTIL_SHIFT_ELEMENT_HPP_
#define BOOM_CPPUTIL_SHIFT_ELEMENT_HPP_

#include <vector>
#include "cpputil/report_error.hpp"

namespace BOOM {

  // Shift an element of a vector to a new position in the same vector, moving
  // intervening elements as needed.
  //
  // Args:
  //   v:  The vector to be manipulated.
  //   from:  The index of the element to be moved.
  //   to:  The new location of the element to be shifted.
  //
  // Example:
  //   std::vector<int> v = {0, 1, 2, 3};
  //   shift_element(v, 2, 0);  // v = {2, 0, 1, 3}
  template <class C>
  void shift_element(std::vector<C> &v, int from, int to) {
    if (from < 0 || to < 0 || from >= v.size() || to >= v.size()) {
      report_error("Illegal arguments to shift_element.");
    }
    if (from == to) return;
    if (from > to) {
      v.insert(v.begin() + to, v[from]);
      v.erase(v.begin() + from + 1);
    } else {
      v.insert(v.begin() + to + 1, v[from]);
      v.erase(v.begin() + from);
    }
  }

}  // namespace BOOM

#endif  //  BOOM_CPPUTIL_SHIFT_ELEMENT_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2014 Steven L. Scott

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

#ifndef BOOM_MAKE_UNIQUE_PRESERVE_ORDER_HPP_
#define BOOM_MAKE_UNIQUE_PRESERVE_ORDER_HPP_

#include <set>
#include <vector>

namespace BOOM {

  // Args:
  //   input: A vector of objects for which operators == and < are
  //     both defined.
  // Returns:
  //   The vector 'input' is copied to the return value.  All but the
  //   first obserations of multiple elements removed.  The order of
  //   elements is otherwise preserved.
  //
  // This is a more expensive operation than calling std::unique on a
  // sorted range.  Use it only when the objects in input are cheap to
  // copy and compare.
  template <class T>
  std::vector<T> make_unique_preserve_order(const std::vector<T> &input) {
    std::set<T> already_seen;
    std::vector<T> ans;
    ans.reserve(input.size());

    for (int i = 0; i < input.size(); ++i) {
      const T &element(input[i]);
      if (already_seen.find(element) == already_seen.end()) {
        ans.push_back(element);
        already_seen.insert(element);
      }
    }
    return ans;
  }

}  // namespace BOOM

#endif  //  BOOM_MAKE_UNIQUE_PRESERVE_ORDER_HPP_

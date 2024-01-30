#ifndef BOOM_CPPUTIL_FIND_HPP_
#define BOOM_CPPUTIL_FIND_HPP_

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

#include <vector>
#include <cstdint>
#include "uint.hpp"
#include "math/Permutation.hpp"

#include <sstream>

namespace BOOM {

  template <class T>
  std::string print_vector(const std::vector<T> &x) {
    std::ostringstream out;
    for (const auto &el : x) {
      out << el << ' ';
    }
    out << "\n";
    return out.str();
  }


  // Return the indices (in sorted_target) of a set of items contained in
  // sorted_input.
  //
  // Args:
  //   sorted_input:  A set of items to search for.
  //   sorted_target: A set of items in which to search.  Both arguments are
  //     sorted in ascending order.
  //
  // Returns:
  //   If every element of sorted_input appears in sorted_target then a vector
  //   of the same size as sorted input is returned.  If an element in
  //   sorted_input is missing from sorted_target then the returned vector will
  template<class T, class U>
  std::vector<Int> find_sorted(
      const std::vector<T> &sorted_input,
      const std::vector<U> &sorted_target) {
    std::vector<Int> ans;
    // The cursor points to a value of sorted_target;
    Int cursor = 0;
    for (Int i = 0; i < sorted_input.size(); ++i) {
      if (i > 0 && sorted_input[i] == sorted_input[i-1]) {
        // Handle 'runs' of the same input data.
        //
        // When searching for [1, 2, 2, 3] in [1, 2, 2, 2, 2, 3] the output should
        // be [0, 1, 2, 5]
        ++cursor;
      }
      while (sorted_target[cursor] < sorted_input[i]
             && cursor < sorted_target.size()) {
        ++cursor;
      }
      if (cursor == sorted_target.size()) {
        return ans;
      } else if (sorted_target[cursor] == sorted_input[i]) {
        ans.push_back(cursor);
      } else {
        ans.push_back(-1);
      }
    }
    return ans;
  }

  // Find the index in a target array of each element of an input array.
  // Elements that cannot be found are assigned a position of -1.
  //
  // Args:
  //   input:  The set of elements to search for.
  //   target:  The set of elements in which to conduct the search.
  //
  // Returns:
  //   A vector (ans) of the same length as 'input', where ans[i] is the
  //   position in 'target' occupied by element i of input.
  template <class T>
  std::vector<Int> find(const std::vector<T> &input,
                        const std::vector<T> &target) {

    Permutation<Int> input_order = Permutation<Int>::order(input);
    Permutation<Int> target_order = Permutation<Int>::order(target);

    std::vector<Int> positions = find_sorted<T, T>(
        input_order * input,
        target_order * target);

    // Now 'positions' contains the locations in the sorted target vector.  We
    // need to map those positions back to their locations in the unsored
    // vector.
    std::vector<Int> ans;
    ans.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
      // Notice that the forward permutation 'target_order' is used here, but
      // the inverse permutation is used below.
      if (positions[i] >=0) {
        ans.push_back(target_order[positions[i]]);
      } else {
        ans.push_back(-1);
      }
    }

    // Now those positions need to be moved to their unsorted positions in the
    // input vector.
    return input_order.inverse() * ans;
  }


}  // namespace BOOM

#endif  //  BOOM_CPPUTIL_FIND_HPP_

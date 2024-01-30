#ifndef CPPUTIL_INDEX_TABLE_H
#define CPPUTIL_INDEX_TABLE_H

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

#include <algorithm>
#include <vector>
#include "uint.hpp"

namespace BOOM {
  // ------------------ NOTE ----------------------------------------
  // New code should probably not use an index table, but the more easily
  // remembered 'Permutation' found in math/Permutation.hpp.

  // An 'index_table' is a sequence of numbers indx such that v[indx[i]] <=
  // v[indx[i+1]].  It is the equivalent of R's 'order' or numpy's 'argsort'.
  //
  // Args:
  //   input:  The vector to be sorted.
  // Returns:
  //    The index table with elements of type INT.
  template <class OBJ, class INT=Int>
  std::vector<INT> index_table(const std::vector<OBJ> &input) {
    std::vector<INT> ans(input.size());
    for (INT i = 0; i < input.size(); ++i) ans[i] = i;
    std::sort(ans.begin(), ans.end(),
              [&input](INT I, INT J){
                return input[I] < input[J];
              });
    return ans;
  }

}  // namespace BOOM

#endif  // CPPUTIL_INDEX_TABLE_H

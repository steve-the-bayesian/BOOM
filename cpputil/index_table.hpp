#ifndef CREATE_INDEX_TABLE_H
#define CREATE_INDEX_TABLE_H

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
  // ------------------ NOTE ----------------------------------------

  // An 'index_table' is a sequence of numbers indx such that v[indx[i]] <=
  // v[indx[i+1]].  It is the equivalent of R's 'order' or numpy's 'argsort'.

  // The inverse of an index_table is a rank_table which gives each
  // observation's rank in the vector v.  E.g. if v[i] < v[j] then
  // rank_table[i] < rank_table[j].

  template <class OBJ, class INT=Int>
  class index_table_less {
    const std::vector<OBJ> &V;

   public:
    explicit index_table_less(const std::vector<OBJ> &v) : V(v) {}
    bool operator()(const INT &i, const INT &j) const { return V[i] < V[j]; }
  };

  template <class OBJ, class INT=Int>
  std::vector<INT> index_table(const std::vector<OBJ> &v) {
    index_table_less<OBJ> Less(v);
    std::vector<INT> ans(v.size());
    for (INT i = 0; i < v.size(); ++i) ans[i] = i;
    std::sort(ans.begin(), ans.end(), Less);
    return ans;
  }

  template <class OBJ, class INT=Int>
  std::vector<INT> index_table(const std::vector<OBJ> &v,
                               double (*val)(const OBJ &)) {
    typedef typename std::vector<OBJ>::size_type sz;
    std::vector<double> vec(v.size());
    for (sz i = 0; i < v.size(); ++i) vec[i] = (*val)(v[i]);
    return index_table(vec);
  }

  template <class OBJ, class INT=Int>
  std::vector<INT> rank_table(const std::vector<OBJ> &v) {
    std::vector<INT> indx = index_table(v);
    std::vector<INT> ans(indx.size());
    for (typename std::vector<INT>::size_type i = 0; i < indx.size(); ++i)
      ans[indx[i]] = i;
    return ans;
  }

  // Compute a rank table from a vector of objects and a function pointer that
  // assigns each a numeric value.
  template <class OBJ, class INT=Int>
  std::vector<INT> rank_table(const std::vector<OBJ> &v,
                              double (*val)(const OBJ &)) {
    typedef typename std::vector<OBJ>::size_type sz;
    std::vector<double> vec(v.size());
    for (sz i = 0; i < v.size(); ++i) {
      vec[i] = (*val)(v[i]);
    }
    return rank_table(vec);
  }

}  // namespace BOOM
#endif  // CREATE_INDEX_TABLE_H

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

#ifndef CREATE_INDEX_TABLE_H
#define CREATE_INDEX_TABLE_H
#include <algorithm>
#include <vector>

namespace BOOM {

  // An 'index_table' is a sequence of numbers indx such that v[indx[i]]
  // <= v[indx[i+1]].

  // The inverse of an index_table is a rank_table which gives each
  // observation's rank in the vector v.  E.g. if v[i] < v[j] then
  // rank_table[i] < rank_table[j].

  template <class OBJ>
  class index_table_less {
    const std::vector<OBJ> &V;

   public:
    explicit index_table_less(const std::vector<OBJ> &v) : V(v) {}
    bool operator()(const int &i, const int &j) const { return V[i] < V[j]; }
  };

  template <class OBJ>
  std::vector<int> index_table(const std::vector<OBJ> &v) {
    index_table_less<OBJ> Less(v);
    std::vector<int> ans(v.size());
    for (int i = 0; i < v.size(); ++i) ans[i] = i;
    std::sort(ans.begin(), ans.end(), Less);
    return ans;
  }

  template <class OBJ>
  std::vector<int> index_table(const std::vector<OBJ> &v,
                               double (*val)(const OBJ &)) {
    // returns an STL vector of integers indx such that
    // v[indx[i]]<= v[indx[i+1]] with respect to the function val(v[i])

    typedef typename std::vector<OBJ>::size_type sz;
    std::vector<double> vec(v.size());
    for (sz i = 0; i < v.size(); ++i) vec[i] = (*val)(v[i]);
    return index_table(vec);
  }

  template <class OBJ>
  std::vector<int> rank_table(const std::vector<OBJ> &v) {
    std::vector<int> indx = index_table(v);
    std::vector<int> ans(indx.size());
    for (std::vector<int>::size_type i = 0; i < indx.size(); ++i)
      ans[indx[i]] = i;
    return ans;
  }

  template <class OBJ>
  std::vector<int> rank_table(const std::vector<OBJ> &v,
                              double (*val)(const OBJ &)) {
    typedef typename std::vector<OBJ>::size_type sz;
    std::vector<double> vec(v.size());
    for (sz i = 0; i < v.size(); ++i) vec[i] = (*val)(v[i]);
    return rank_table(vec);
  }

}  // namespace BOOM
#endif  // CREATE_INDEX_TABLE_H

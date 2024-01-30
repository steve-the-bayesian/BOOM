// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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
#include "cpputil/apply_permutation.hpp"

namespace BOOM {

  template <class VEC, class INT=int>
  void apply_permutation_impl(const std::vector<INT> &permutation, VEC &data) {
    INT stride = data.stride();
    INT n = data.size();
    INT pk;

    for (INT i = 0; i < n; ++i) {
      INT k = permutation[i];
      while (k > i) k = permutation[k];
      if (k < i) continue;
      // Now have k == i, i.e the least in its cycle
      pk = permutation[k];
      if (pk == i) continue;
      // shuffle the elements of the cycle
      {
        double tmp_data = data[i * stride];

        while (pk != i) {
          {
            double r1 = data[pk * stride];
            data[k * stride] = r1;
          }
          k = pk;
          pk = permutation[k];
        };

        data[k * stride] = tmp_data;
      }
    }
  }

  void permute_inplace(const std::vector<Int> &permutation, Vector &data) {
    apply_permutation_impl<Vector, Int>(permutation, data);
  }

  void permute_inplace(const std::vector<Int> &permutation, VectorView &data) {
    apply_permutation_impl<VectorView, Int>(permutation, data);
  }

  Vector apply_permutation(const std::vector<Int> &permutation,
                           const Vector &data) {
    Vector ans(data);
    permute_inplace(permutation, ans);
    return ans;
  }

  Vector apply_permutation(const std::vector<Int> &permutation,
                           const ConstVectorView &data) {
    Vector ans(data);
    permute_inplace(permutation, ans);
    return ans;
  }

  void permute_inplace(const std::vector<int> &permutation, Vector &data) {
    apply_permutation_impl<Vector, int>(permutation, data);
  }

  void permute_inplace(const std::vector<int> &permutation, VectorView &data) {
    apply_permutation_impl<VectorView, int>(permutation, data);
  }

  Vector apply_permutation(const std::vector<int> &permutation,
                           const Vector &data) {
    Vector ans(data);
    permute_inplace(permutation, ans);
    return ans;
  }

  Vector apply_permutation(const std::vector<int> &permutation,
                           const ConstVectorView &data) {
    Vector ans(data);
    permute_inplace(permutation, ans);
    return ans;
  }

}  // namespace BOOM

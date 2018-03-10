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

#ifndef BOOM_CPPUTIL_APPLY_PERMUTATION_HPP_
#define BOOM_CPPUTIL_APPLY_PERMUTATION_HPP_
#include <vector>
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

namespace BOOM {

  void permute_inplace(const std::vector<int> &permutation, Vector &data);
  void permute_inplace(const std::vector<int> &permutation, VectorView &data);
  Vector apply_permutation(const std::vector<int> &permutation,
                           const Vector &data);
  Vector apply_permutation(const std::vector<int> &permutation,
                           const ConstVectorView &data);

}  // namespace BOOM
#endif  //  BOOM_CPPUTIL_APPLY_PERMUTATION_HPP_

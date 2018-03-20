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

#ifndef BOOM_COMPARE_VECTOR_BOOL_HPP
#define BOOM_COMPARE_VECTOR_BOOL_HPP
#include <vector>
namespace BOOM {
  /*------------------------------------------------------------
     compare vector<bool>'s as binary representations of decimal
     integers.  x[n] is the 2^n place (i.e. x[0] is the unit's place)
     ------------------------------------------------------------*/

  bool less(const std::vector<bool> &x, const std::vector<bool> &y,
            bool or_equal);

  bool less(const std::vector<bool> &x, const std::vector<bool> &y);

  inline bool operator<(const std::vector<bool> &x,
                        const std::vector<bool> &y) {
    return less(x, y);
  }
  bool operator==(const std::vector<bool> &x, const std::vector<bool> &y);
  inline bool operator<=(const std::vector<bool> &x,
                         const std::vector<bool> &y) {
    return less(x, y, false);
  }

  inline bool operator>(const std::vector<bool> &x,
                        const std::vector<bool> &y) {
    return !(x <= y);
  }
  inline bool operator>=(const std::vector<bool> &x,
                         const std::vector<bool> &y) {
    return !(x < y);
  }

  inline bool operator!=(const std::vector<bool> &x,
                         const std::vector<bool> &y) {
    return !(x == y);
  }

}  // namespace BOOM
#endif  // BOOM_COMPARE_VECTOR_BOOL_HPP

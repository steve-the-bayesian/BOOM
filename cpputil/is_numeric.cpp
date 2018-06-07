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

#include <cctype>
#include <string>
#include "uint.hpp"

namespace BOOM {

  inline bool is_e(char c) { return (c == 'e' || c == 'E'); }
  inline bool is_dot(char c) { return (c == '.'); }
  inline bool is_sign(char c) { return (c == '-' || c == '+'); }

  bool is_numeric(const std::string &s) {
    // if all characters in s could be part of a numerical object
    // return true.  If any cannot return false.

    unsigned ndot = 0;
    unsigned ne = 0;
    unsigned ndigits = 0;
    bool last_was_e = false;
    for (uint i = 0; i < s.size(); ++i) {
      char c = s[i];
      if (last_was_e && !is_sign(c)) return false;

      if (is_e(c)) {
        ++ne;
        if (ne > 1) return false;
        last_was_e = true;
        continue;
      } else if (is_dot(c)) {
        ++ndot;
        if (ndot > 1) return false;
      } else if (is_sign(c)) {
        if (i > 0 && last_was_e == false) return false;
      } else if (!isdigit(c)) {
        return false;
      } else {
        ++ndigits;
      }
      last_was_e = false;
    }
    return ndigits > 0;
  }
}  // namespace BOOM

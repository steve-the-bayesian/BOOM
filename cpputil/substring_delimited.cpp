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

#include <string>
namespace BOOM {
  using std::string;
  string substring_delimited(string &s, char delim) {
    string::size_type n = s.find(delim);
    if (n == string::npos) {
      // delim was not found in string s
      return s;
    }

    string ans = s.substr(0, n);
    s = s.substr(n + 1);  // the +1 throws away the delimiter
    return ans;
  }
}  // namespace BOOM

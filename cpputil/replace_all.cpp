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

#include <cstring>
#include <string>
namespace BOOM {

  using std::string;
  string replace_all(const string &s, const char *c1, const char *c2) {
    // replaces all instances of c1 with c2;

    string::size_type n = 0;
    string::size_type n1 = strlen(c1);
    string ans(s);
    while (n != string::npos) {
      n = ans.find(c1);
      if (n != string::npos) ans.replace(n, n1, c2);
    }
    return ans;
  }

  string &replace_all(string &s, const char *c1, const char *c2) {
    // replaces all instances of c1 with c2;

    string::size_type n = 0;
    string::size_type n1 = strlen(c1);
    string &ans(s);
    while (n != string::npos) {
      n = ans.find(c1);
      if (n != string::npos) ans.replace(n, n1, c2);
    }
    return ans;
  }

}  // namespace BOOM

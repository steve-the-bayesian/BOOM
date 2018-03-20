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

#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>

namespace BOOM {
  char *get_date(char *s) {
    char *sp;
    time_t tm;
    tm = time(&tm);

    sprintf(s, "%s", ctime(&tm));
    sp = strstr(s, "\n");
    sp[0] = '\0';
    return s;
  }

  using std::string;

  string get_date() {
    time_t tm;
    tm = time(&tm);
    string s(ctime(&tm));
    string::size_type n = s.find_last_of("\n");
    if (n != string::npos) s.erase(n);
    return s;
  }
}  // namespace BOOM

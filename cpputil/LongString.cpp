// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#include "cpputil/LongString.hpp"
#include <iostream>

namespace BOOM {

  using std::endl;
  using std::ostream;
  using std::string;

  LongString::LongString(const string &str, unsigned w, unsigned p,
                         bool pad_first_line)
      : s(str), width(w), pad(p), pad_first(pad_first_line) {
    width -= pad;
  }

  ostream &LongString::print(ostream &out) const {
    unsigned start = 0;
    unsigned back = s.size();
    string blanks(pad, ' ');
    while (back - start > width) {
      unsigned pos = start + width;
      while (pos > start) {
        if (s[pos] == ' ') break;
        --pos;
      }
      if (pos == start) pos = start + width;
      if (start > 0 || (start == 0 && pad_first)) out << blanks;
      out << s.substr(start, pos - start) << endl;
      start = pos + 1;
      while (start < back && s[start] == ' ') ++start;
    }
    if (back - start > 0) {
      if (start > 0 || (start == 0 && pad_first)) out << blanks;
      out << s.substr(start, back - start) << endl;
    }
    return out;
  }
}  // namespace BOOM

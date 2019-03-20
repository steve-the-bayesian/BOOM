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

#include <string>
#include <vector>

namespace BOOM {
  std::vector<std::string> split_delimited(
      const std::string &s,
      const std::string &delims) {
    std::vector<std::string> ans;
    typedef std::string::size_type sz;
    sz b = 0;
    bool done = false;

    while (!done) {
      sz e = s.find_first_of(delims, b);
      if (e == std::string::npos) {
        done = true;
        ans.push_back(s.substr(b));
      } else {
        ans.push_back(s.substr(b, e - b));
      }
      b = e + 1;
    }
    return ans;
  }
}  // namespace BOOM

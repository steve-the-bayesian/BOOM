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

#include <cassert>
#include <string>
#include <vector>

namespace BOOM {
  
  std::vector<std::string> split_string(const std::string &s) {
    std::vector<std::string> ans;
    const std::string ws(" \n\r\t\f\v");

    std::string::size_type b = s.find_first_not_of(ws);
    if (b == std::string::npos) return ans;

    while (1) {
      std::string::size_type e = s.find_first_of(ws, b);
      assert(e >= b);
      if (e == std::string::npos) {
        ans.push_back(s.substr(b));
        return ans;
      } else {
        ans.push_back(s.substr(b, e - b));
        b = s.find_first_not_of(ws, e);
        if (b == std::string::npos) return ans;
      }
    }
  }
}  // namespace BOOM

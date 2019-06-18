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
#include <fstream>
#include <string>
#include <vector>

namespace BOOM {

  std::vector<std::string> read_file(std::istream &in) {
    std::vector<std::string> ans;
    while (in) {
      std::string line;
      getline(in, line);
      if (!in) break;
      ans.push_back(line);
    }
    return ans;
  }

  std::vector<std::string> read_file(const std::string &fname) {
    std::ifstream in(fname.c_str());
    return read_file(in);
  }
}  // namespace BOOM

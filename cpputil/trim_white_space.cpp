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
#include <string>
#include <vector>
namespace BOOM {

  std::string trim_white_space(const std::string &s) {
    typedef std::string::size_type sz;
    sz b = s.find_first_not_of(" \n\t\f\r\v");
    if (b == std::string::npos) return std::string("");
    sz e = s.find_last_not_of(" \n\t\f\r\v");
    //    if(b==e) return std::string("");
    return s.substr(b, 1 + (e - b));
  }

  void trim_white_space(std::vector<std::string> &v) {
    unsigned int n = v.size();
    for (unsigned int i = 0; i < n; ++i) v[i] = trim_white_space(v[i]);
  }

}  // namespace BOOM

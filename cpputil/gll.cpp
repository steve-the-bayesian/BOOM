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
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/

#include <fstream>
#include <ios>
#include <iostream>
namespace BOOM {

  typedef std::basic_ios<char>::pos_type pos_type;

  std::ifstream &gll(std::ifstream &in) {
    // sets in so that it points to the last line of a data file

    in.seekg(0, std::ios_base::beg);
    pos_type begin = in.tellg();

    in.seekg(-1, std::ios_base::end);
    pos_type end = in.tellg();

    if (begin == end) return in;

    char c = in.peek();
    while ((c == '\n' || c == '\r') && in.tellg() > 0) {
      in.seekg(-1, std::ios_base::cur);
      c = in.peek();
    }
    do {
      in.seekg(-1, std::ios_base::cur);
      c = in.peek();
    } while (c != '\n' && c != '\r' && in.tellg() > 0);
    pos_type now = in.tellg();
    if (now != end && now != begin) in.get(c);  // read in last newline
    return in;
  }

}  // namespace BOOM

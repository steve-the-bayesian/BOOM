// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#ifndef BOOM_TO_STRING_HPP_
#define BOOM_TO_STRING_HPP_

#include <sstream>
#include <string>

namespace BOOM {
  // Convert "object" to a string using its streaming operator.  This is a
  // useful utility to have when streaming is inconvenient (e.g. when working
  // with a debugger or a third party logging system that doesn't support
  // streaming.)
  template <class T>
  std::string ToString(const T &object) {
    std::ostringstream out;
    out << object;
    return out.str();
  }

  template <class T>
  std::string ToString(const std::vector<T> &vector) {
    std::ostringstream out;
    for (size_t i = 0; i < vector.size(); ++i) {
      out << vector[i];
      if (i + 1 < vector.size()) {
        out << " ";
      }
    }
    return out.str();
  }

}  // namespace BOOM

#endif  //  BOOM_TO_STRING_HPP_

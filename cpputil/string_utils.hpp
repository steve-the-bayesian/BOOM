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

#ifndef CPP_STRING_UTILS_H
#define CPP_STRING_UTILS_H
#include <string>
#include <vector>
#include <ostream>
#include "uint.hpp"
#include "cpputil/Split.hpp"
#include "uint.hpp"

namespace BOOM {

  std::vector<std::string> split_string(const std::string &);
  std::vector<std::string> split_delimited(const std::string &s,
                                           const std::string &delim);
  inline std::vector<std::string> split_delimited(const std::string &s,
                                                  char delim) {
    return split_delimited(s, std::string(1, delim));
  }

  inline std::vector<std::string> split(const std::string &s) {
    return split_string(s);
  }
  inline std::vector<std::string> split(const std::string &s,
                                        const std::string &d) {
    return split_delimited(s, d);
  }
  inline std::vector<std::string> split(const std::string &s, char dlm) {
    return split_delimited(s, dlm);
  }

  inline std::ostream &operator<<(
      std::ostream &out,
      const std::vector<std::string> &string_vector) {
    for (const auto &s : string_vector) {
      out << s;
      out << " ";
    }
    return out;
  }
  
  std::ostream &print_columns(
      std::ostream &out,
      const std::vector<std::vector<std::string>> &columns,
      uint pad = 2);

  std::ostream &print_two_columns(
      std::ostream &out, const std::vector<std::string> &left,
      const std::vector<std::string> &right,
      uint pad = 2);

  bool is_all_white(const std::string &s);

  // Removes all white space.
  std::string strip_white_space(const std::string &s);

  // Remove white space from the front and back.
  std::string trim_white_space(const std::string &s);   
  void trim_white_space(std::vector<std::string> &v);

  std::string strip(const std::string &s, const std::string &bad = "\r\n\t");
  // removes \r's, \n's etc from end

  std::string replace_all(const std::string &s, const char *, const char *);
  std::string &replace_all(std::string &s, const char *, const char *);

  bool is_numeric(const std::string &s);

  // Concatenate the contents in a vector of strings to a single string.
  //
  // Args:
  //   string_vector:  The strings to concatenate.
  //   sep:  A separator to follow all but the last element.
  //
  // Returns:
  //   A string formed by concatenating the elements of string_vector.
  std::string concatenate(const std::vector<std::string> &string_vector,
                          const std::string &sep = " ");
  
}  // namespace BOOM

#endif  // CPP_STRING_UTILS_H

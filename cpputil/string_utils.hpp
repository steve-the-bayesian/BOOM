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
#include "BOOM.hpp"
#include "cpputil/Split.hpp"
#include "uint.hpp"

namespace BOOM {
  std::vector<string> split_delimited(const string &s, const string &delim);
  std::vector<string> split_string(const string &);

  inline std::vector<string> split_delimited(const string &s, char delim) {
    return split_delimited(s, string(1, delim));
  }
  inline std::vector<string> split(const string &s) { return split_string(s); }
  inline std::vector<string> split(const string &s, const string &d) {
    return split_delimited(s, d);
  }
  inline std::vector<string> split(const string &s, char dlm) {
    return split_delimited(s, dlm);
  }

  inline ostream &operator<<(ostream &out, const std::vector<std::string> &sv) {
    for (uint i = 0; i < sv.size(); ++i) out << sv[i] << " ";
    return out;
  }

  ostream &print_columns(ostream &out,
                         const std::vector<std::vector<std::string>> &columns,
                         uint pad = 2);

  ostream &print_two_columns(ostream &out, const std::vector<std::string> &left,
                             const std::vector<std::string> &right,
                             uint pad = 2);

  string operator+(const std::string &, int);
  string operator+(const std::string &, double);
  string operator+=(string &, int);
  string operator+=(string &, double);
  string operator+(int, const std::string &);
  string operator+(double, const std::string &);

  string operator>>(const std::string &, int &);
  string operator>>(const std::string &, double &);

  bool is_all_white(const string &s);
  string strip_white_space(const string &s);  // removes all white space
  string trim_white_space(const string &s);   // removes from the ends
  void trim_white_space(std::vector<std::string> &v);

  string strip(const string &s, const string &bad = "\r\n\t");
  // removes \r's, \n's etc from end

  string replace_all(const string &s, const char *, const char *);
  string &replace_all(string &s, const char *, const char *);

  inline char last(const string &s) { return s[s.length() - 1]; }
  inline char &last(string &s) { return s[s.length() - 1]; }

  bool is_numeric(const string &s);
}  // namespace BOOM
#endif  // CPP_STRING_UTILS_H

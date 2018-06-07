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

#ifndef BOOM_CPP_FILE_UTILS_H
#define BOOM_CPP_FILE_UTILS_H

#include <vector>
#include "uint.hpp"
#include "cpputil/gll.hpp"
#include "uint.hpp"

namespace BOOM {

  uint count_lines(const std::string &fname);
  std::string add_to_path(const std::string &path, const std::string &s);
  void legalize_file_name(std::string &);

  std::string pwd();
  std::string get_dpath(const std::string &);
  std::string strip_path(const std::string &fname);
  bool check_directory(const std::string &);
  void mkdir(const std::string &);
  void check_empty(const std::string &dir);
  std::vector<string> read_file(const string &fname);
  std::vector<string> read_file(istream &);

}  // namespace BOOM
#endif  // BOOM_CPP_FILE_UTILS_H

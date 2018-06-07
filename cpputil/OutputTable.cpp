// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2008 Steven L. Scott

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
#include "cpputil/OutputTable.hpp"

namespace BOOM {
  typedef OutputTable OT;

  OT::OutputTable(uint pad) : pad_(pad) {}

  std::vector<std::string> &OT::column(uint i) { return cols_[i]; }

  OT &OT::add_column(const std::vector<std::string> &col) {
    cols_.push_back(col);
    return *this;
  }

  OT &OT::add_to_column(const std::string &s, uint i) {
    cols_[i].push_back(s);
    return *this;
  }

  OT &OT::add_row(const std::vector<std::string> &row) {
    equalize_rows();
    uint rl = row.size();
    for (uint i = 0; i < rl; ++i) cols_[i].push_back(row[i]);
    return *this;
  }

  void OT::equalize_rows() {
    uint nr = 0;
    uint nc = cols_.size();
    for (uint i = 0; i < nc; ++i) nr = std::max<uint>(nr, cols_[i].size());
    for (uint i = 0; i < nc; ++i)
      while (cols_[i].size() < nr) cols_[i].push_back("");
  }

  std::ostream &OT::print(std::ostream &out) const {
    return print_columns(out, cols_, pad_);
  }

}  // namespace BOOM

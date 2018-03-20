// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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

#include "cpputil/parse_range.hpp"
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include "cpputil/report_error.hpp"
#include "cpputil/seq.hpp"
#include "uint.hpp"

namespace BOOM {
  using namespace std;
  using std::string;
  typedef RangeParser RP;

  vector<unsigned> parse_range(const string &s) {
    RP rp;
    return rp(s);
  }

  RP::RangeParser() : not_found(string::npos) {}

  vector<unsigned int> RP::operator()(const string &s) {
    range = s;
    check_range();
    ans.clear();
    while (!range.empty()) {
      find_block();
      parse_block();
    }
    return ans;
  }

  void RP::find_block() {
    sz comma_pos = range.find(',');
    if (comma_pos == not_found) {
      block = range;
      range.clear();
    } else {
      // separate parts and throw away comma
      block = range.substr(0, comma_pos);
      range = range.substr(comma_pos + 1);
    }
  }

  void RP::parse_block() {
    sz dash_pos = block.find('-');
    if (dash_pos == not_found) {
      uint number = atoi(block.c_str());
      ans.push_back(number);
    } else {
      istringstream in(block);
      char dash;
      uint from, to;
      in >> from >> dash >> to;
      vector<uint> irng = seq(from, to);
      std::copy(irng.begin(), irng.end(), back_inserter(ans));
    }
  }

  void RP::check_range() {
    sz bad = range.find_first_not_of("0123456789,-");
    if (bad == not_found) return;
    ostringstream msg;
    msg << "Illegal characters passed to RangeParser(string) : " << range
        << std::endl
        << " only positive integers, commas (,) , and dashes (-) allowed.";
    report_error(msg.str());
  }

}  // namespace BOOM

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
#ifndef BOOM_PARSE_RANGE_HPP
#define BOOM_PARSE_RANGE_HPP

#include <string>
#include <vector>

namespace BOOM {

  // returns a

  std::vector<unsigned int> parse_range(const std::string &s);

  class RangeParser {
   public:
    typedef std::string::size_type sz;
    RangeParser();
    std::vector<unsigned int> operator()(const std::string &);

   private:
    void check_range();
    void find_block();
    void parse_block();
    std::string range;
    sz not_found;
    std::vector<unsigned int> ans;
    std::string block;
  };

}  // namespace BOOM
#endif  // BOOM_PARSE_RANGE_HPP

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
#ifndef BOOM_STRING_SPLIT_HPP
#define BOOM_STRING_SPLIT_HPP

#include <string>
#include <vector>

namespace BOOM {
  class StringSplitter {
   public:
    // The default splitter uses either spaces or tabs to separate fields.
    explicit StringSplitter(const std::string &sep = " \t",
                            bool allow_quotes = true);

    // Split the string 's' into a vector of strings.
    std::vector<std::string> operator()(const std::string &s) const;

   private:
    // The set of characters used as field delimiters.
    std::string delim;

    // The set of characters to interpret as quotes.
    std::string quotes;

    // The splitter is delimited if it uses something other than white space to
    // separate fields.
    bool delimited;
  };

}  // namespace BOOM

#endif  // BOOM_STRING_SPLIT_HPP

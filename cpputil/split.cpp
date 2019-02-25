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
#include "cpputil/Split.hpp"
#include <boost/tokenizer.hpp>
#include <cctype>
#include <string>
#include <vector>
#include "cpputil/report_error.hpp"
#include "cpputil/string_utils.hpp"

namespace BOOM {
  using std::string;
  using std::vector;

  StringSplitter::StringSplitter(const string &s, bool allow_quotes)
      : delim(s),
        quotes(allow_quotes ? "\"'" : ""),
        delimited(!is_all_white(s)) {}

  //------------------------------------------------------------
  std::vector<std::string> StringSplitter::operator()(const string &s) const {
    typedef boost::escaped_list_separator<char> Sep;
    typedef boost::tokenizer<Sep> tokenizer;

    Sep sep("", delim, quotes);
    tokenizer tk(s, sep);
    if (delimited) {
      return std::vector<std::string>(tk.begin(), tk.end());
    }

    std::vector<std::string> ans;
    for (tokenizer::iterator it = tk.begin(); it != tk.end(); ++it) {
      string token = *it;
      if (!token.empty()) ans.push_back(token);
      if (ans.size() > s.size()) {
        std::ostringstream err;
        err << "Too many fields discovered in the string " << s;
        report_error(err.str());
      }
    }
    return ans;
  }

}  // namespace BOOM

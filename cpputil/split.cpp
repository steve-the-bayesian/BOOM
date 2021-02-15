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
#include <cctype>
#include <string>
#include <vector>
#include "cpputil/report_error.hpp"
#include "cpputil/string_utils.hpp"

namespace BOOM {
  using std::string;
  using std::vector;

  StringSplitter::StringSplitter(const string &delimiters, bool allow_quotes)
      : delim_(delimiters),
        quotes_(allow_quotes ? "\"'" : ""),
        delimited_(!is_all_white(delimiters)) {}

  std::vector<std::string> StringSplitter::operator()(const std::string &s) const {
    if (delimited_) {
      return split_delimited(s);
    } else {
      return split_space(s);
    }
  }

  std::vector<std::string> StringSplitter::split_delimited(const std::string &s) const {
    std::vector<std::string> ans;
    if (s.empty()) {
      return ans;
    }
    const char *start = s.data();
    const char *end = start + s.size();
    // Test cases
    // 3,
    // ,,
    //
    while(start != end) {
      while (is_field_delimiter(*start)) {
        ans.push_back("");
        ++start;
        if (start == end) {
          ans.push_back("");
          return ans;
        }
      }
      const char *pos = find_field_boundary(start, end);
      std::string field(start, pos);
      ans.push_back(strip_quotes(field));
      if (is_field_delimiter(*pos) && pos + 1 == end) {
        ans.push_back("");
      }
      start = std::min(pos + 1, end);
    }
    return ans;
  }

  std::vector<std::string> StringSplitter::split_space(const std::string &s) const {
    std::vector<std::string> ans;
    const char *start = s.data();
    const char *end = start + s.size();
    // test cases: "1 2 3     4 5   "
    // ""
    // " "
    // "    1    "
    // "1"
    // "1 "
    // " 1"
    // "1 2"
    // " 1 2"
    // "1 2 "
    while (start != end) {
      // There's more to read.
      while(*start == ' ' && start != end) {
        // Move past initial white space.
        ++start;
      }
      if (start == end) {
        return ans;
      }
      char open_quote = ' ';
      if (is_quote(*start)) {
        open_quote = *start;
      }
      const char *pos = find_whitespace(start, end, open_quote);
      std::string field(start, pos);
      ans.push_back(strip_quotes(field));
      start = pos;
    }
    return ans;
  }

  std::string StringSplitter::strip_quotes(const string &s) const {
    if (allow_quotes()) {
      if (s.size() >= 2 && is_quote(s[0]) && s.back() == s[0]) {
        return s.substr(1, s.size() - 2);
      }
    }
    return s;
  }

  bool StringSplitter::inside_field(const char *pos, const char *end,
                                    char &open_quote) const {
    if (pos == end) return false;
    if (open_quote != ' ' && allow_quotes()) {
      // If we are in the middle of a quoted string, check to see if the quote
      // is closed.  If so, set the state
      if (*pos == open_quote) {
        open_quote = ' ';
      }
      // Inside the quoted part of a field, so return true.
      return true;
    }

    // If we encounter a quote and we're not in the middled of a quoted string
    // then remember the quote (to mark the beginning of a quoted string), and
    // return false.
    if (is_quote(*pos) && allow_quotes()) {
      open_quote = *pos;
      // At the opening quote for a field.
      return true;
    }

    // If we find a field delimiter then we're outside a field.  Otherwise we're
    // inside.
    return !is_field_delimiter(*pos);
  }

  // Returns a pointer to the character one position after the end of the field.
  const char *StringSplitter::find_field_boundary(const char *start, const char *end) const {
    // blah, blah, blah,,,foo, bar, baz
    //
    const char *pos = start;
    char open_quote = ' ';
    while (inside_field(++pos, end, open_quote)) {
      // Do nothing.
    }
    return pos;
  }

  const char *StringSplitter::find_whitespace(const char *start,
                                              const char *end,
                                              char &open_quote) const {
    const char *pos = start;
    while(inside_field(++pos, end, open_quote)) {}
    return pos;
  }


  bool StringSplitter::is_field_delimiter(char c) const {
    return delim_.find(c) != std::string::npos;
  }

}  // namespace BOOM

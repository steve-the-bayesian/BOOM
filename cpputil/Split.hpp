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

  // Split input strings by delimiter or white space.
  class StringSplitter {
   public:
    // Args:
    //   sep: The field separator.  When the parser encounters any character in
    //     'sep' a new field is generated.  The default splitter uses either
    //     spaces to separate fields.
    //   allow_quotes: If true then separators enclosed in quotes do not result
    //     in new fields.
    explicit StringSplitter(const std::string &sep = " ",
                            bool allow_quotes = true);

    // Split the string 's' into a vector of strings.
    std::vector<std::string> operator()(const std::string &s) const;

   private:
    // Returns 'true' if quoted fields are allowed, false otherwise.
    bool allow_quotes() const {
      return !quotes_.empty();
    }

    // Split string into fields separated by delimiters.
    std::vector<std::string> split_delimited(const std::string &s) const;

    // Split string into fields separated by one or more spaces.
    std::vector<std::string> split_space(const std::string &s) const;

    // If quotes are enabled and if the leading and trailing characters in 's'
    // are matching quotes then remove them and return the string inside.
    // Otherwise return s.
    std::string strip_quotes(const std::string &s) const;

    // True if c is a recognized quote character, false otherwise.
    bool is_quote(const char c) const {
      return quotes_.find(c) != std::string::npos;
    }

    // True if c is inside a field.
    // Args:
    //   pos:  The current position in the string, to be checked.
    //   end:  The end of the string, which is outside of any field.
    //   open_quote: If quotes are allowed and pos follows an open quote that
    //     has yet to be closed, then 'open_quote' is the type of opening quote
    //     that was encountered.  If not in the midst of a quoted field, then
    //     open_quote should be a space ' '.  If *pos is a quote then open_quote
    //     will be changed to *pos.
    //
    // Returns:
    //   True if *pos is not a field delimiter, and false if it is.
    bool inside_field(const char *pos,
                      const char *end,
                      char &open_quote) const;

    // Returns the first position after the end of the specified field.
    // Args:
    //   start: Points to the beginning of the current field.
    //   end:  The end of the buffer.
    // Returns:
    //   A pointer to one past the end of the current field, or end, if no
    //   intervening field is found.
    const char *find_field_boundary(const char *start, const char *end) const;

    // Find the first space following non-space.
    // Args:
    //   start:  The beginning of the substring to check.  *start must not be a space.
    //   end:  One past the end of the string.
    //   open_quote: If quotes are allowed and pos follows an open quote that
    //     has yet to be closed, then 'open_quote' is the type of opening quote
    //     that was encountered.  If not in the midst of a quoted field, then
    //     open_quote should be a space ' '.  If *pos is a quote then open_quote
    //     will be changed to *pos.
    const char *find_whitespace(const char *start, const char *end,
                                char &open_quote) const;

    // Returns true iff c is one of the characters listed in 'delim_'.
    bool is_field_delimiter(char c) const;

    //---------------------------------------------------------------------------
    // Data Section

    // The set of characters used as field delimiters.
    std::string delim_;

    // The set of characters to interpret as quotes.
    std::string quotes_;

    // The splitter is delimited if it uses something other than white space to
    // separate fields.
    bool delimited_;
  };

}  // namespace BOOM

#endif  // BOOM_STRING_SPLIT_HPP

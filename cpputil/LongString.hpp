// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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
#ifndef BOOM_LONGSTRING_PRINTER_HPP
#define BOOM_LONGSTRING_PRINTER_HPP

#include <iosfwd>
#include <string>
namespace BOOM {

  class LongString {
    /*
     *  Class to control the printing of very long strings.
     */
   public:
    explicit LongString(const std::string &str, unsigned Width = 80, unsigned Pad = 0,
               bool pad_first_line = true);
    std::ostream &print(std::ostream &out) const;

   private:
    std::string s;
    unsigned width, pad;
    bool pad_first;
  };

  inline std::ostream &operator<<(std::ostream &out, const LongString &s) {
    return s.print(out);
  }

}  // namespace BOOM
#endif  // BOOM_LONGSTRING_PRINTER_HPP

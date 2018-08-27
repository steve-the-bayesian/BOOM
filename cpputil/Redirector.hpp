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
#ifndef BOOM_STREAM_REDIRECTOR
#define BOOM_STREAM_REDIRECTOR

#include <iosfwd>

namespace BOOM {
  // Redirect the output from one stream to another.
  //
  // Example:
  // std::ostringstream out;
  // {
  //    Redirector redirect(std::cout, out);
  //    std::cout << "foo bar baz";
  // }
  // std::cout << out.str();
  class Redirector {
   public:
    Redirector(std::ostream &from, std::ostream &to);
    ~Redirector();

   private:
    std::streambuf *from_buf_, *to_buf_;
    std::ostream *from_;
  };
  
}  // namespace BOOM
#endif  // BOOM_STREAM_REDIRECTOR

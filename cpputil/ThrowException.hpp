// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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
#ifndef BOOM_THROW_EXCEPTION_HPP_
#define BOOM_THROW_EXCEPTION_HPP_

#include <stdexcept>
#include <string>

namespace BOOM {
#ifdef NO_BOOM_EXCEPTIONS
  // exception handling function in namespace boom supplied by the user
  void nothrow_exception(const std::string &s);
#endif

  template <class EXCEPTION>
  void throw_exception(const std::string &s) {
#ifndef NO_BOOM_EXCEPTIONS
    throw EXCEPTION(s);
#else
    nothrow_exception(s);
#endif
  }

}  // namespace BOOM
#endif  // BOOM_THROW_EXCEPTION_HPP_

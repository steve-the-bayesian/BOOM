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
#include "cpputil/report_error.hpp"
#include <iostream>
#include "cpputil/ThrowException.hpp"


#ifdef RLANGUAGE
#include <Rinternals.h>
#endif

namespace BOOM {
  void report_error(const std::string &msg) {
    throw_exception<std::runtime_error>(msg);
  }

  void report_warning(const std::string &msg) {
#ifdef RLANGUAGE
    Rf_warning("%s\n", msg.c_str());
#else
    std::cerr << "Warning:  " << msg << std::endl;
#endif
  }

  void report_message(const std::string &msg) {
#ifdef RLANGUAGE
    Rprintf("%s\n", msg.c_str());
#else
    std::cout << msg << std::endl;
#endif
  }

}  // namespace BOOM

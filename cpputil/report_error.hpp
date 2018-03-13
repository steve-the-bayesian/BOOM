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
#ifndef BOOM_REPORT_ERROR_HPP_
#define BOOM_REPORT_ERROR_HPP_
#include <iomanip>
#include <sstream>
#include <string>

namespace BOOM {

  // Code that calls report_error typically also includes
  // ostringstream to build the error message and uses iomanipulators
  // like setw and endl, so it makes sense to add these into the BOOM
  // namespace here.

  using std::endl;
  using std::ifstream;
  using std::ios;
  using std::iostream;
  using std::istream;
  using std::istringstream;
  using std::ofstream;
  using std::ostream;
  using std::ostringstream;
  using std::setw;

  // The usual method of reporting an error is to throw a
  // std::runtime_error with the given msg as its payload.
  void report_error(const std::string &msg);
  inline void report_error(const std::ostringstream &message_buffer) {
    report_error(message_buffer.str());
  }

  // The usual method of reporting a warning is to write to stderr.
  void report_warning(const std::string &msg);
  inline void report_warning(const std::ostringstream &message_buffer) {
    report_warning(message_buffer.str());
  }

  // The usual method of reporting a message is to write to stdout.
  void report_message(const std::string &msg);

  inline void report_message(const std::ostringstream &message_buffer) {
    report_message(message_buffer.str());
  }

}  // namespace BOOM

#endif  // BOOM_REPORT_ERROR_HPP_

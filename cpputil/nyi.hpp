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

#ifndef BOOM_NYI_HPP
#define BOOM_NYI_HPP

#include <sstream>
#include "cpputil/report_error.hpp"

namespace BOOM {

  inline void nyi(const std::string& thing) {
    std::ostringstream err;
    err << thing << " is not yet implemented.\n";
    report_error(err.str());
  }
}  // namespace BOOM
#endif  // BOOM_NYI_HPP

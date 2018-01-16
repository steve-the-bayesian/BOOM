/*
  Copyright (C) 2007-2014 Steven L. Scott

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
#ifndef BOOM_PORTABLE_MATH_HPP_
#define BOOM_PORTABLE_MATH_HPP_

// For platforms that do not implement the full cmath library required
// by the standard, pull the implementation from boost into namespace
// std.

#ifdef _MSC_VER
#include <boost/math/special_functions/fpclassify.hpp>
namespace std{
  using boost::math::isnan;
}
#endif  // _MSC_VER

#ifdef __sun
// Provide versions of isnan and isfinite that solaris chooses not to
// provide.
#include <boost/math/special_functions/fpclassify.hpp>
namespace std {
  using boost::math::isnan;
  using boost::math::isfinite;
}
#endif  // __sun

#include <boost/math/special_functions/round.hpp>
namespace BOOM {
using boost::math::lround;
}

#endif  // BOOM_PORTABLE_MATH_HPP_

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

#ifndef BOOM_MATH_CONSTANTS_HPP_
#define BOOM_MATH_CONSTANTS_HPP_
namespace BOOM {

  struct Constants {
    static constexpr double pi = 3.141592653589793;
    static constexpr double pi_squared = 9.86960440108936;
    static constexpr double half_pi_squared = 4.93480220054468;
    static constexpr double root_2pi = 0.918938533204673;
    static constexpr double log_root_2pi = 2.506628274631;
    static constexpr double log_pi = 1.1447298858494;
  };

}  // namespace BOOM
#endif  // BOOM_MATH_CONSTANTS_HPP_

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
#ifndef BOOM_SCALAR_NEWTON_MAX_HPP_
#define BOOM_SCALAR_NEWTON_MAX_HPP_
#include "TargetFun/TargetFun.hpp"

namespace BOOM {

  // newton-raphson routine to maximize the given target.  the value
  // of the function at the max is returned, x is set to the
  // maximizing value.  g and h are the first and second derivatives
  // at the max (g should be close to zero, and h should be negative).
  double scalar_newton_max(const d2ScalarTargetFun &f, double &x, double &g,
                           double &h);
}  // namespace BOOM
#endif  // BOOM_SCALAR_NEWTON_MAX_HPP_

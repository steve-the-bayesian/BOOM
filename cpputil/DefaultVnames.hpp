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
#ifndef BOOM_DEFAULT_VNAMES_HPP
#define BOOM_DEFAULT_VNAMES_HPP

#include <string>
#include <vector>
#include "uint.hpp"

namespace BOOM {

  // produces default names for a set of variables of a given size

  std::vector<std::string> default_vnames(uint sz, int indx_start = 0);
}  // namespace BOOM
#endif  // BOOM_DEFAULT_VNAMES_HPP

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/Policies/NullParamPolicy.hpp"
namespace BOOM {
  typedef NullParamPolicy NPP;

  NPP::NullParamPolicy() {}

  NPP::NullParamPolicy(const NPP &rhs) : Model(rhs) {}

  NPP &NPP::operator=(const NPP &) { return *this; }

  ParamVector NPP::parameter_vector() { return ParamVector(); }
  const ParamVector NPP::parameter_vector() const { return ParamVector(); }
}  // namespace BOOM

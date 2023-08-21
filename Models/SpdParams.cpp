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
#include "Models/SpdParams.hpp"
namespace BOOM {

  typedef SpdParams SP;
  typedef SpdData SD;

  SP::SpdParams(uint p, double diag, bool ivar) : SD(p, diag, ivar) {}

  SP::SpdParams(const SpdMatrix &V, bool ivar) : SD(V, ivar) {}

  SP *SP::clone() const { return new SP(*this); }

  Vector SP::vectorize(bool min) const { return var().vectorize(min); }

  Vector::const_iterator SP::unvectorize(Vector::const_iterator &v,
                                         bool minimal) {
    SpdMatrix tmp(var());
    Vector::const_iterator ans = tmp.unvectorize(v, minimal);
    set_var(tmp);
    return ans;
  }

}  // namespace BOOM

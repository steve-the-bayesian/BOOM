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

#include "LinAlg/Matrix.hpp"
#include "numopt.hpp"

namespace BOOM {

  double Negate::operator()(const Vector &x) const { return -1 * f(x); }
  double dNegate::operator()(const Vector &x, Vector &g) const {
    double ans = df(x, g);
    g *= -1;
    return -1 * ans;
  }
  double d2Negate::operator()(const Vector &x, Vector &g, Matrix &h) const {
    double ans = d2f(x, g, h);
    g *= -1;
    h *= -1.0;
    return -1 * ans;
  }

}  // namespace BOOM

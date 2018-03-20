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
#ifndef BOOM_HMM_TOOLS_HPP
#define BOOM_HMM_TOOLS_HPP

#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {

  double fwd_1(Vector &pi, Matrix &P, const Matrix &logQ, const Vector &logd,
               const Vector &one);
  void bkwd_1(Vector &pi, Matrix &P, Vector &wsp, const Vector &one);

}  // namespace BOOM
#endif  // BOOM_HMM_TOOLS_HPP

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

#ifndef BOOM_DRAW_LOGIT_LAMBDA_HPP_
#define BOOM_DRAW_LOGIT_LAMBDA_HPP_
#include "distributions.hpp"
namespace BOOM {

  namespace Logit {
    double draw_lambda_mt(RNG& rng, double r);
    bool check_right(double u, double lam);
    bool check_left(double u, double lam);
  }  // namespace Logit
}  // namespace BOOM
#endif  // BOOM_DRAW_LOGIT_LAMBDA_HPP_

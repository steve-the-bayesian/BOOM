// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2009 Steven L. Scott

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

#include "distributions/rng.hpp"
#include <ctime>
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  RNG::RNG()
      : generator_(std::random_device()())
  {}

  RNG::RNG(RngIntType seed)
      : generator_(seed)
  {}

  void RNG::seed() {
    generator_.seed(std::random_device()());
  }

  RNG::RngIntType seed_rng(RNG &rng) {
    RNG::RngIntType ans = 0;
    while (ans <= 2) {
      double u = runif_mt(rng) * static_cast<double>(
          std::numeric_limits<RNG::RngIntType>::max());
      ans = lround(u);
    }
    return ans;
  }

  RNG GlobalRng::rng(8675309);

}  // namespace BOOM

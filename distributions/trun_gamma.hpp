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

#ifndef BOOM_TRUN_GAMMA_HPP
#define BOOM_TRUN_GAMMA_HPP

#include "distributions/rng.hpp"
#include "distributions.hpp"

namespace BOOM {
  // Density of the truncated Gamma(a,b) distribution with support >= cut.
  //
  // Args:
  //   x:  The argument of the density function.
  //   a:  The shape parameter of the gamma distribution.
  //   b: The scale parameter of the distribution.  The mean of the distribution
  //      is a/ b
  //   cut: The beginning of the support of the distribution.
  //   logscale: If true, return the log density.  If false return the density.
  //   normalize: If true then the density is normalized.  If false it is not.
  //     Note that the unnormalized density is proportional to the density of an
  //     un-truncated gamma distribution.
  //
  // Returns:
  //   The (log) density value at x.
  double dtrun_gamma(double x, double a, double b, double cut,
                     bool logscale, bool normalize=false);

  // Returns a draw from the truncated Gamma(a,b) distribution with
  // support >= cut.
  //
  // Args:
  //   rng:  The U(0, 1) random number generator to use.
  //   a:  The shape parameter of the gamma distribution.
  //   b: The scale parameter of the distribution.  The mean of the distribution
  //      is a/ b
  //   cut: The beginning of the support of the distribution.
  //   nslice: If a slice sampler is used to simulate the draws, how many slice
  //     sampling iterations should be used.  Larger numbers lead to greater
  //     independence, but unnecessary iterations are expensive.
  //
  // Returns:
  //   A draw x from the truncated Gamma(a, b) distribution conditional on x >=
  //   cut.
  double rtrun_gamma(double a, double b, double cut, unsigned nslice = 5);
  double rtrun_gamma_mt(RNG &, double a, double b, double cut, unsigned nslice = 5);
}  // namespace BOOM
#endif  // BOOM_TRUN_GAMMA_HPP

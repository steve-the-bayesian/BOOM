// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2009 Steven L. Scott

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
#ifndef BOOM_INVERSE_GAUSSIAN_HPP_
#define BOOM_INVERSE_GAUSSIAN_HPP_
#include "distributions.hpp"

namespace BOOM {

  // Functions to evaluate the density and the CDF of the inverse
  // Gaussian distribution, and to take random draws from the
  // distribution.
  //
  // The inverse Gaussian distribution for a random variable x with
  // parameters lambda and mu has density function
  //
  //   f(x) =  (lambda / 2 \pi x^3)^{1/2}
  //               * exp( -lambda (x - mu)^2 / 2 mu^2 x)
  //
  // The mean of this distribution is mu.  The mean of a reciprocal
  // deviate is (1/mu) + (1/lambda).
  //
  // The variance of this distribution is mu^3 / lambda. The variance
  // of a reciprocal deviate is (1 / mu * lambda) + (2 / lambda^2).

  double dig(double x, double mu, double lambda, bool logscale);
  double pig(double x, double mu, double lambda, bool logscale);
  double rig_mt(RNG& rng, double mu, double lambda);
  inline double rig(double mu, double lambda) {
    return rig_mt(GlobalRng::rng, mu, lambda);
  }

}  // namespace BOOM
#endif  // BOOM_INVERSE_GAUSSIAN_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

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

#ifndef BOOM_T_DATA_IMPUTER_HPP_
#define BOOM_T_DATA_IMPUTER_HPP_

#include "distributions/rng.hpp"

namespace BOOM {

  // A draw y from the T distribution with nu "degrees of freedom" can
  // be written
  //   y | mu, w,s,nu ~ mu + N(0, s^2) / sqrt(w)
  //            w ~ Ga(nu/2, nu/2)
  //
  //  p(w) \propto  w^(nu/2 - 1) exp(-(nu/2)*w)
  //  p(y | w) \propto  (w/s^2)^(1/2) exp(- 0.5 *w * (y-mu)^2 / s^2)
  //
  //  So by Bayes' rule,
  //  p(w | y) \propto w^((nu + 1)/2 - 1) exp(-w  * (nu + (y-mu)^2/s^2)/2)
  //           \propto Ga( (nu+1)/2, (nu + (y-mu)^2/s^2)/2 )
  //
  // This class provides the imputation logic to impute the latent w
  // given y-mu, s^2, and nu.
  //
  // If mu is a normal mean or a linear combination of regression
  // coefficients then w can be used as a weight in a weighted least
  // squares regression with y as a response variable.
  class TDataImputer {
   public:
    // Args:
    //   rng:  A random number generator.
    //   residual:  y-mu in the comment above.
    //   sd:  s in the comment above.
    //   nu: nu in the comment above (the "degrees of freedom" parameter).
    // Returns:
    //   A random draw of w from its posterior distribution.
    double impute(RNG &rng, double residual, double sd, double df) const;
  };

}  // namespace BOOM
#endif  //  BOOM_T_DATA_IMPUTER_HPP_

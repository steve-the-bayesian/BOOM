#ifndef BOOM_MIXED_DATA_IMPUTER_HPP_
#define BOOM_MIXED_DATA_IMPUTER_HPP_
/*
  Copyright (C) 2005-2020 Steven L. Scott

  This library is free software; you can redistribute it and/or modify it under
  the terms of the GNU Lesser General Public License as published by the Free
  Software Foundation; either version 2.1 of the License, or (at your option)
  any later version.

  This library is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
  details.

  You should have received a copy of the GNU Lesser General Public License along
  with this library; if not, write to the Free Software Foundation, Inc., 51
  Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA
*/

namespace BOOM {


  // A model responsible for imputing missing values and correcting likely
  // errors.
  //
  // The model has components for categorical and numeric varibles, with joint
  // distribution p(categorical) * p(numeric | categorical).  The categorical
  // component is a "pattern matching" mixture model.  The categorical variables
  // V1, ..., Vm are modeled as conditionally product multinomial given a
  // cluster indcator z.  That is, the conditional probability that V1 = v1,
  // ... Vm = vm given z is pi_1(z)[v1] * ... * pi_m(z)[vm].
  //
  // The conditional distribution of the numeric variables given the categorical
  // variables is a transformed multivariate regression.  If the vector of
  // numeric variables is Y, and the dummy variable expansion of the categorical
  // variables is X, then there is a transformation h such that h(Y) ~ N(X*beta,
  // Sigma).  The transformation we use is the Gaussian copula: h(Y_j) =
  // Phi^{-1}(F_j(Y_j)), where F_j is the empirical CDF of variable j.  This
  // produces h(Y) values that are marginally standard normal.
  //
  //
  class MixedDataImputer {

  };


}  // namespace BOOM

#endif  // BOOM_MIXED_DATA_IMPUTER_HPP_

// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2011 Steven L. Scott

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

namespace BOOM {

  // Compute the probability that binomial proportion 1, which has
  // produced successes1 successes out of trials1 trials is less than
  // binomial proportion2.  Both proportions are a-priori independent
  // with a common Beta(prior_successes, prior_failures) prior
  // distribution.
  double compare_binomial_proportions(double success1, double success2,
                                      double trials1, double trials2,
                                      double prior_successes = 1.0,
                                      double prior_failures = 1.0);

}  // namespace BOOM

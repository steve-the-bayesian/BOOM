// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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
#ifndef BOOM_MARKOV_DIST_HPP
#define BOOM_MARKOV_DIST_HPP

#include <vector>
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "uint.hpp"

namespace BOOM {
  class Selector;
  using BOOM::uint;

  // Return the stationary distribution of the transition matrix Q.
  // Each row of Q sums to 1.
  Vector get_stat_dist(const Matrix &Q);

  // Return the probability that state r happens before state s in a Markov
  // chain with initial distribution pi0 and transition matrix P.
  double preceeds(uint r, uint s, const Vector &pi0, const Matrix &P);

  // Return the probability that any of the states in r happen before any of the
  // states in s in a Markov chain with initial distribution pi0 and transition
  // matrix P
  double preceeds(const Selector &r, const Selector &s, const Vector &pi0,
                  const Matrix &P);

  // Compute the conditional probability of being absorbed into each absorbing
  // state, given a current value in a particular transient state.
  //
  // Args:
  //   P:  The transition probability matrix.
  //   abs: A selector with size matching P, indicating which states are
  //     absorbing states.
  //
  // Returns:
  //   A matrix with rows corresponding to transient states, and columns to
  //   absorbing states.  S-|abs| rows and |abs| columns.  Each row is a
  //   probability distribution giving the conditional probability of being
  //   absorbed into a particular state.
  Matrix compute_conditional_absorption_probs(const Matrix &P,
                                              const Selector &abs);
}  // namespace BOOM
#endif  // BOOM_MARKOV_DIST_HPP

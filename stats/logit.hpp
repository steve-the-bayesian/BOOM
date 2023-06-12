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
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301,
   USA
 */

#ifndef STATS_LOGIT_HPP
#define STATS_LOGIT_HPP

#include <cmath>
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"
#include "distributions/Rmath_dist.hpp"

namespace BOOM {

  // Convert from the probabliity scale to the logit (log odds) scale.
  inline double logit(double prob) { return qlogis(prob); }

  // Convert from the logit (log odds) scale to the probabliity scale.
  inline double logit_inv(double logit) { return plogis(logit); }

  // Apply the logit transformation to a vector of probabilities.
  // Args:
  //   probs: A vector of probabilities, each giving the chance of a different
  //     event.
  // Returns:
  //   A vector of the same length as 'probs', giving event chances on the logit
  //   scale.
  inline Vector logit(const Vector &probs) {
    Vector ans(probs);
    for (size_t i = 0; i < ans.size(); ++i) {
      ans[i] = logit(double(ans[i]));
    }
    return ans;
  }

  // Element-by-element application of the inverse logit function.
  // Args:
  //   logits: A vector of real numbers, each representing the chance of an
  //     event on the logit scale.
  // Returns:
  //   The probability associated with each element of the argument, computed
  //   using the inverse logit transformation.
  inline Vector logit_inv(const Vector &logits) {
    Vector ans(logits);
    for (size_t i = 0; i < ans.size(); ++i) ans[i] = logit_inv(ans[i]);
    return ans;
  }

  // "Log one plus exp" of x.
  // Args:
  //   x:  A real number.
  // Returns:
  //   log(1 + exp(x)), with care to avoid overflow or underflow.
  inline double lope(double x) {
    if (x > 0)
      return x + ::log1p(exp(-x));
    else
      return ::log1p(exp(x));
  }

  // Args:
  //   distribution: A vector of non-negative numbers summing to 1.  The last
  //     element should be positive.
  // Returns:
  //   A vector of dimension one smaller than the argument.  Element i is
  //     log(distribution[i] / distribution.back());
  Vector multinomial_logit(const Vector &distribution);
  Vector multinomial_logit(const VectorView &distribution);
  Vector multinomial_logit(const ConstVectorView &distribution);

  // The inverse of the multinomial_logit transformation.
  // Args:
  //   logits: A vector of real numbers.
  // Returns:
  //   A vector of dimension one larger than the argument.  Element [i] contains
  //   exp(logits[i]) / sum(exp(logits[i])).
  //
  // Care is taken to avoid overflow from the exponential.
  Vector multinomial_logit_inverse(const Vector &logits);
  Vector multinomial_logit_inverse(const VectorView &logits);
  Vector multinomial_logit_inverse(const ConstVectorView &logits);

}  // namespace BOOM

#endif  // STATS_LOGIT_HPP

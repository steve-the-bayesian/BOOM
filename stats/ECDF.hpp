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
#ifndef BOOM_STATS_EMPIRICAL_CDF_HPP_
#define BOOM_STATS_EMPIRICAL_CDF_HPP_

#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

namespace BOOM {

  // The empirical cumulative distribution function of a real-valued data set.
  class ECDF {
   public:
    // An empty ecdf cannot be evaluated.
    ECDF() {}

    // Args:
    //   unsorted:  The data set.
    explicit ECDF(const ConstVectorView &unsorted_data);

    ECDF(const ECDF &rhs) = default;
    ECDF(ECDF &&rhs) = default;
    ECDF &operator=(const ECDF &rhs) = default;
    ECDF &operator=(ECDF &&rhs) = default;

    // The fraction of the data less than or equal to x.
    double fplus(double x) const;

    // The fraction of the data strictly less than x.
    double fminus(double x) const;

    // Args:
    //   x: potential value of a random variable.  The argument of the
    //     distribution function.
    //   equality: If true (the default) then the return is Pr(X <= x).  If
    //     false then the return is Pr(X < x).
    //
    // Returns:
    //   The empirical probability that a randomly chosen observation is <= x
    //   (if 'equality' is true), or < x (otherwise).
    double operator()(double x, bool equality = true) const {
      return equality ? fplus(x) : fminus(x);
    }

    // The quantile function, which is the inverse of the CDF.  Returns the
    // number that corresponds to a given probability.
    //
    // Args:
    //   probability:  The probabliity for which a quantile is desired.
    //
    // Returns:
    //   The number appearing 'probability' percent of the way through the
    //   distribution.
    double quantile(double probability) const;

    const Vector &sorted_data() const { return sorted_data_; }

    void restore(const Vector &sorted_data) {sorted_data_ = sorted_data;}

   private:
    Vector sorted_data_;
  };
}  // namespace BOOM

#endif  // BOOM_STATS_EMPIRICAL_CDF_HPP_

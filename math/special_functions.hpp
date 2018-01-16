/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#ifndef BOOM_MATH_SPECIAL_FUNCTIONS_HPP_
#define BOOM_MATH_SPECIAL_FUNCTIONS_HPP_

namespace BOOM {

  namespace Cephes {
    double polylog(int n, double x);
    double spence(double x);
  }

  //======================================================================
  // Special functions from the cephes math library:
  //
  // The polylog function, denoted Li_n(x) is
  //  Li_n(x) = \sum_{k = 1}^\infy x^k / k^n
  //
  // When n == 2 this is the called the dilog function.  It shows up
  // when computing moments for the truncated logistic distribution.
  inline double polylog(int n, double x) {
    return Cephes::polylog(n, x);
  }

  // The dilogarithm.  Equivalent to polylog(2, x), but faster for x
  // in (0,1).
  inline double dilog(double x) {
    if (0 < x && x < 1) {
      return Cephes::spence(1 - x);
    } else {
      return polylog(2, x);
    }
  }

  //======================================================================
  // Returns the log of the multivariate gamma function.
  // Args:
  //   x: The primary function argument.
  //   dimension: The subscript of the multivariate gamma function,
  //     often denoted p or d.
  //
  // Returns:
  //   The log of
  //   Gamma_p(x) = pi^(p(p-1) / 4) * \prod_{i-1}^p Gamma(x + (1-i) /2)
  double lmultigamma(double x, int dimension);

  // Returns the log of the ratio Gamma_p(x + extra/2) / Gamma_p(x) for
  // integer values of extra.  If 'extra' < 'dimension' then
  // significant cancellation can occur.
  // Args:
  //   x: The argument of the bottom function.
  //   extra:  The increment to the argument of the top function.  >= 0.
  //   dimension:  The subscript of the multivariate gamma function.
  double lmultigamma_ratio(double x, int extra, int dimension);

}  // namespace BOOM
#endif // BOOM_MATH_SPECIAL_FUNCTIONS_HPP_

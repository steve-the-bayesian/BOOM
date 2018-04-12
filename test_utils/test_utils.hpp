#ifndef BOOM_TEST_UTILS_HPP_
#define BOOM_TEST_UTILS_HPP_

/*
  Copyright (C) 2018 Steven L. Scott

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

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"

namespace BOOM {

  // A check to see if v1 and v2 are equal to within some tolerance.
  // Args:
  //   v1:  The first vector-like object.
  //   v2:  The second vector-like object.
  //   tol:  The maximum allowable difference, element-wise, between v1 and v2.
  // Returns:
  //   True if all elements of v1 and v2 are within 'tol' of each other.  False
  //   otherwise.
  template <class V1, class V2>
  bool VectorEquals(const V1 &v1, const V2 &v2, double tol = 1e-8) {
    Vector v = v1 - v2;
    return v.max_abs() < tol;
  }

  // A check to see if m1 and m2 are equal to within some tolerance.
  // Args:
  //   m1:  The first matrix-like object.
  //   m22:  The second matrix-like object.
  //   tol:  The maximum allowable difference, element-wise, between m1 and m2.
  // Returns:
  //   True if all elements of m1 and m2 are within 'tol' of each other.  False
  //   otherwise.
  template <class M1, class M2>
  bool MatrixEquals(const M1 &m1, const M2 &m2, double tol = 1e-8) {
    Matrix m = m1 - m2;
    return m.max_abs() < tol;
  }

  // The data structure returned from check_mcmc_matrix().
  struct CheckMatrixStatus {
    CheckMatrixStatus()
        : ok(true),
          fails_to_cover(0),
          fraction_failing_to_cover(0),
          failure_rate_limit(0)
    {}

    // A human readable error message that should be examined in case 'ok' is
    // false.
    std::string error_message() const;
    
    // The primary return type.  True iff 'draws' cover 'truth' acceptably well.
    bool ok;

    // The number of times 'draws' failed to cover 'truth' at the specified
    // confidence.
    int fails_to_cover;

    double fraction_failing_to_cover;
    double failure_rate_limit;
  };

  // Printing a status object prints its error message.
  inline ostream & operator<<(ostream &out, const CheckMatrixStatus &status) {
    out << status.error_message();
    return(out);
  }
  
  // Check to see if a matrix of Monte Carlo draws covers a known set of true
  // values acceptably well.
  //
  // Args:
  //   draws: A matrix of Monte Carlo draws to be checked.  Each row is a draw
  //     and each column is a variable.
  //   truth: A vector of true values against which draws will be compared.
  //     truth.size() must match ncol(draws).
  //   confidence: Used for two things.  First, each column of 'draws' is used
  //     to construct a Monte Carlo central credibility interval with
  //     probability content 'confidence.'  Second, around 'confidence' percent
  //     of the intervals must cover the true values.
  //   control_multiple_comparisons: If true, then the fraction of intervals
  //     successfully covering the true value can be slightly less than
  //     'confidence' as long as it is consistent with the null hypothesis of
  //     'confidence'-level coverage, in the sense that it is no less than two
  //     binomial standard errors below 'confidence'.
  //
  // Returns:
  //   A status message with the value 'ok' indicating whether the check passed
  //   (ok == true) or failed (ok == false).  The remainder of the status
  //   message is there to help with useful error messages.
  CheckMatrixStatus check_mcmc_matrix(const Matrix &draws,
                                      const Vector &truth,
                                      double confidence = .95,
                                      bool control_multiple_comparisons = true);

  // Check to see if a vector of Monte Carlo draws covers a known value.
  //
  // Args:
  //   draws:  The vector of Monte Carlo draws to check.
  //   truth:  The true value against which 'draws' will be checked.
  //   confidence: The probability content of the credibility interval used to
  //     check coverage.
  //
  // Returns:
  //   A central credibility interval with probability content 'confidence' is
  //   constructed from 'draws'.  The boolean return indicates whether this
  //   interval covers 'truth'.
  bool CheckMcmcVector(const Vector &draws,
                       double truth,
                       double confidence = .95);

  // Returns true if the empirical CDF of the vector of data matches the
  // theoretical CDF to with the tolerance of a Kolmogorov-Smirnoff test.
  // Args:
  //   data: A sample of data ostensibly from the cdf passed as the second
  //     argument.
  //   cdf:  The mathematical CDF ostensibly responsible for the data.
  //   significance: The significance level of the KS test comparing 'data' to
  //     'cdf'.
  //
  // Returns:
  //   An empirical CDF of data is computed and used to implement a KS test.
  //   The return value is true if the p-value of the text exceeds the given
  //   significance level.
  bool DistributionsMatch(const Vector &data,
                          const std::function<double(double)> &cdf,
                          double significance = .05);
}  // namespace BOOM

#endif //  BOOM_TEST_UTILS_HPP_

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

  //===========================================================================
  // The data structure returned from check_mcmc_matrix().
  struct CheckMatrixStatus {
    CheckMatrixStatus()
        : ok(true),
          fails_to_cover(0),
          fraction_failing_to_cover(0),
          failure_rate_limit(0),
          dimension_mismatch(false)
    {}

    // A human readable error message that should be examined in case 'ok' is
    // false.
    std::string error_message() const;

    // The primary return type.  True iff 'draws' cover 'truth' acceptably well.
    bool ok;

    // The number of times 'draws' failed to cover 'truth' at the specified
    // confidence.
    int fails_to_cover;

    // The fraction of draws that did not cover the true value.
    double fraction_failing_to_cover;

    // The maximum value fraction_failing_to_cover can have before we declare
    // a test failure.
    double failure_rate_limit;

    // Indicates that the "truth" value passed into CheckMcmcMatrix did not
    // conform.
    bool dimension_mismatch;
  };

  // Printing a status object prints its error message.
  inline std::ostream & operator<<(std::ostream &out,
                                   const CheckMatrixStatus &status) {
    out << status.error_message();
    return(out);
  }

  //===========================================================================
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
  //   filename: The name of a file to which the matrix will be printed if the
  //     check fails.  The first row of the file will be the true value.  If the
  //     file name is the empty string no file will be created.
  //
  // Returns:
  //   A status message with the value 'ok' indicating whether the check passed
  //   (ok == true) or failed (ok == false).  The remainder of the status
  //   message is there to help with useful error messages.
  CheckMatrixStatus CheckMcmcMatrix(const Matrix &draws,
                                    const Vector &truth,
                                    double confidence = .95,
                                    bool control_multiple_comparisons = true,
                                    const std::string &filename = "");

  // A check similar to CheckMcmcMatrix, but designed for stochastic processes
  // or other functions exhibiting serial correlation, which can mess up the
  // multiple comparisons adjustments used by CheckMcmcMatrix.
  //
  // Args:
  //   draws: A matrix of Monte Carlo draws to be checked.  Each row is a draw
  //     and each column is a variable.
  //   truth: A vector of true values against which draws will be compared.
  //     truth.size() must match ncol(draws).
  //   confidence: The confidence associated with the marginal posterior
  //     intervals used to determine coverage.
  //   sd_ratio_threshold: One of the testing diagnostics compares the standard
  //     deviation of the centered draws to the standard deviation of the true
  //     function.  If that ratio is less than this threshold the diagnostic is
  //     passed.
  //   coverage_fraction: The fraction of marginal posterior intervals that must
  //     cover their true values at the specified confidence level in order for
  //     the check to pass.
  //   filename: The name of a file to which the matrix will be printed if the
  //     check fails.  The first row of the file will be the true value.  If the
  //     file name is the empty string no file will be created.
  //
  // Details:
  //   Half the marginal confidence intervals should cover, and the residual
  //   standard deviation should be small relative to the standard deviation of
  //   truth.
  //
  // Returns:
  //   An error message.  An empty message means the test passed.
  std::string CheckStochasticProcess(const Matrix &draws,
                                     const Vector &truth,
                                     double confidence = .95,
                                     double sd_ratio_threshold = .1,
                                     double coverage_fraction = 0.5,
                                     const std::string &filename = "");

  //===========================================================================
  // A non-empty return value is an error message indicating the first column of
  // 'draws' to fall outside the range [lo, hi].
  std::string CheckWithinRage(const Matrix &draws, const Vector &lo,
                              const Vector &hi);

  // A non-empty return value is an error message indicating the first column of
  // 'draws' to fall outside the range [lo, hi].
  std::string CheckWithinRage(const Vector &draws, double lo, double hi);

  //===========================================================================
  // Check to see if a vector of Monte Carlo draws covers a known value.
  //
  // Args:
  //   draws:  The vector of Monte Carlo draws to check.
  //   truth:  The true value against which 'draws' will be checked.
  //   confidence: The probability content of the credibility interval used to
  //     check coverage.
  //   filename: The name of a file to which the matrix will be printed if the
  //     check fails.  The first row of the file will be the true value.  If the
  //     file name is the empty string no file will be created.
  //   force_file_output: If true then print output to a file even of the check
  //     succeeds.
  //
  // Returns:
  //   A central credibility interval with probability content 'confidence' is
  //   constructed from 'draws'.  The boolean return indicates whether this
  //   interval covers 'truth'.
  bool CheckMcmcVector(const Vector &draws,
                       double truth,
                       double confidence = .95,
                       const std::string &filename = "",
                       bool force_file_output = false);

  // Compare the median of 'draws' as the Y variable with 'truth' as the X
  // variable in a simple regression.  Look for an intercept close to zero, a
  // slope close to 1, and a fairly high R^2.
  //
  // Args:
  //   draws:  Each row is a draw, each column is a different variable.
  //   truth: A Vector with the same length as the number of columns in 'draws'.
  //     These are the true values that 'draws' are supposed to cover.
  //   r2_threshold: The r-square threshold that must be exceeded in order for
  //     the test to pass.
  //
  // Returns:
  //   true if the R^2 between 'truth' and the median of 'draws' equals or
  //   exceeds the specified threshold.  Returns false otherwise.
  bool CheckTrend(const Matrix &draws, const Vector &truth, double r2_threshold);

  //===========================================================================
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

  //===========================================================================
  // Performs a 2-sample Kolmogorov Smirnoff test that the two sets of draws are
  // from the same distribution.
  //
  // Args:
  //   data1, data2: Sets of draws thought to be from the same distribution.
  //   significance:  The significance level of the KS test.
  //
  // Returns:
  //   If the null hypothesis cannot be rejected at the given significance level
  //   then this function returns 'true'.  If the null is rejected then 'false'
  //   is returned.  In other words, 'true' indicates that 'data1' and 'data2'
  //   are a match.
  bool TwoSampleKs(const ConstVectorView &data1,
                   const ConstVectorView &data2,
                   double significance = .05);

  //===========================================================================
  // Checks that two sets of Monte Carlo draws have roughly the same center and
  // spread.  The check is done by computing the .2 - .8 credible interval from
  // one set of draws, and checking that it covers the .3 - .7 interval for the
  // second.  The check is then done in reverse.
  //
  // This check is useful if there is an old, slow, reliable way of sampling
  // from a distribution, and you want to check that the new, fast way gets the
  // same answer.
  //
  // This check is more lenient than the TwoSampleKsMatch check above, but it is
  // also less sensitive to the assumption of independent draws.
  //
  // Args:
  //   draws1, draws2:  Two sets of Monte Carlo draws
  //
  // Returns:
  //   If the distributions are similar (in the sense defined above) then 'true'
  //   is returned.  Otherwise 'false' is returned.
  bool EquivalentSimulations(const ConstVectorView &draws1,
                             const ConstVectorView &draws2);


  // A utility function.
  // Args:
  //   draws: A collection of Monte Carlo draws describing a probability
  //     distribution.
  //   value: A value to compare to the probability distribution in 'draws'.
  //   confidence: The confidence value used to determine if 'value' lies inside
  //     'draws'.
  //
  // Returns true iff the 'confidence' central interval from 'draws' includes
  // 'value'.
  bool covers(const ConstVectorView &draws, double value, double confidence);

}  // namespace BOOM

#endif //  BOOM_TEST_UTILS_HPP_

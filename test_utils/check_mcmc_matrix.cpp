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

#include "test_utils/test_utils.hpp"
#include "cpputil/report_error.hpp"
#include "stats/moments.hpp"
#include <fstream>

namespace BOOM {

  std::string CheckMatrixStatus::error_message() const {
    std::ostringstream err;
    if (dimension_mismatch) {
      err << "The dimension of the 'truth' vector did not match the number of "
          "columns in the matrix being checked.";
    } else {
      err << "Too many columns of 'draws' failed to cover true values." << "\n"
          << "Failure rate: " << fraction_failing_to_cover * 100 << " (%) "
          << "\n"
          << "Rate limit: " << failure_rate_limit * 100 << " (%) " << endl;
    }
    return err.str();
  }

  bool covers(const ConstVectorView &draws, double value, double confidence) {
    double alpha = 1 - confidence;
    Vector sorted = sort(draws);
    double lower = sorted_vector_quantile(sorted, alpha / 2);
    double upper = sorted_vector_quantile(sorted, 1 - (alpha / 2));
    return value >= lower && value <= upper;
  }

  CheckMatrixStatus CheckMcmcMatrix(
      const Matrix &draws,
      const Vector &truth,
      double confidence,
      bool control_multiple_comparisons,
      const std::string &filename) {
    if (confidence <= 0 || confidence >= 1) {
      report_error("Confidence must be strictly between 0 and 1.");
    }
    if (confidence < .5) confidence = 1 - confidence;
    CheckMatrixStatus status;
    if (truth.size() != draws.ncol()) {
      status.ok = false;
      status.dimension_mismatch = true;
    } else {
      for (int i = 0; i < ncol(draws); ++i) {
        if (!covers(draws.col(i), truth[i], confidence)) {
          ++status.fails_to_cover;
        }
      }

      double fraction_failing_to_cover = status.fails_to_cover;
      fraction_failing_to_cover /= ncol(draws);
      double coverage_rate_limit = confidence;
      if (control_multiple_comparisons) {
        double se = sqrt(confidence * (1 - confidence) / ncol(draws));
        coverage_rate_limit -= 2 * se;
      }
      status.failure_rate_limit = 1 - coverage_rate_limit;
      if (fraction_failing_to_cover >= status.failure_rate_limit) {
        status.ok = false;
        status.fraction_failing_to_cover = fraction_failing_to_cover;
      }
    }

    if (!status.ok && filename != "") {
      std::ofstream error_file(filename);
      error_file << truth << std::endl << draws;
    }

    return status;
  }



  std::string CheckWithinRage(const Matrix &draws, const Vector &lo,
                                    const Vector &hi) {
    if (draws.ncol() != lo.size()
        || draws.ncol() != hi.size()) {
      report_error("Both 'lo' and 'hi' must have length equal to the number"
                   " of columns in 'draws'.");
    }
    for (int i = 0; i < draws.ncol(); ++i) {
      double min_draw = min(draws.col(i));
      double max_draw = max(draws.col(i));
      if (hi[i] < lo[i]) {
        std::ostringstream err;
        err << "hi < lo for element " << i
            << ".  Please check the test inputs.";
        report_error(err.str());
      }
      if (min_draw < lo[i] || max_draw > hi[i]) {
        std::ostringstream err;
        err << "The range of column " << i << " was ["
            << min_draw << ", " << max_draw << "], which falls outside of ["
            << lo[i] << ", " << hi[i] << "]." << std::endl;
        return err.str();
      }
    }
    return "";
  }

  std::string CheckWithinRage(const Vector &draws, double lo, double hi) {
    if (hi < lo) {
      report_error("hi must be at least as large as lo.");
    }
    double min_draw = min(draws);
    double max_draw = max(draws);
    if (min_draw < lo || max_draw > hi) {
      std::ostringstream err;
      err << "The range of daws was [" << min_draw
          << ", " << max_draw << "] which falls outside of ["
          << lo << ", " << hi << "].";
      return err.str();
    }
    return "";
  }

  bool CheckMcmcVector(const Vector &draws, double truth, double confidence,
                       const std::string &filename,
                       bool force_file_output) {
    if (confidence <= 0 || confidence >= 1) {
      report_error("Confidence must be strictly between 0 and 1.");
    }
    if (confidence < .5) confidence = 1 - confidence;
    double alpha = 1 - confidence;
    double alpha_2 = .5 * alpha;
    Vector v = sort(draws);
    double lo = sorted_vector_quantile(v, alpha_2);
    double hi = sorted_vector_quantile(v, 1 - alpha_2);
    bool ok = lo <= truth && hi >= truth;
    if (force_file_output || (!ok && filename != "")) {
      std::ofstream error_file(filename);
      error_file << truth << " " << draws;
    }
    return ok;
  }

}  // namespace BOOM

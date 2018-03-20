// Copyright 2011 Google Inc. All Rights Reserved.
// Author: stevescott@google.com (Steve Scott)

#include <algorithm>
#include <vector>
#include "Rinternals.h"
#include "r_interface/boom_r_tools.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SubMatrix.hpp"

extern "C" {
  // Sum the results of a fine-scale time series to a coarser scale.
  // Args:
  //   r_fine_series: An R vector or matrix to aggregate.  If a matrix
  //     is passed, then the columns of the matrix represent time.
  //   r_contains_end: An R logical vector indicating whether each
  //     fine-scale time interval in r_fine_series contains the end of
  //     a coarse time interval.
  //   r_membership_fraction: An R numeric vector containing the
  //     fraction of the output that should be attributed to the
  //     coarse-scale interval containing the beginning of each
  //     fine-scale time interval.  This is always positive, and is
  //     typically 1.
  // Returns:
  //   An R matrix (if r_fine_series is a matrix) or vector
  //   (otherwise) containing the time series aggregation of
  //   r_fine_series.
  //
  //   Note that unless r_fine_series happens to coincide with the
  //   exact beginning or end of a coarse time interval, the left and
  //   right end points of the resulting aggregation will not contain
  //   full aggregates.
  SEXP analysis_common_r_bsts_aggregate_time_series_(
      SEXP r_fine_series,
      SEXP r_contains_end,
      SEXP r_membership_fraction) {
    int *contains_end = LOGICAL(r_contains_end);
    double *membership_fraction = REAL(r_membership_fraction);

    int num_fine_time_points = LENGTH(r_contains_end);
    int num_fine_rows = 1;

    if (Rf_isMatrix(r_fine_series)) {
      num_fine_rows = Rf_nrows(r_fine_series);
    }

    int num_coarse_time_points = 0;
    for (int i = 0; i < num_fine_time_points; ++i) {
      bool end = contains_end[i];
      num_coarse_time_points += end;
    }

    // There is a remainder unless the last entry contains_end and the
    // membership fraction is 1.  The .9999 allows for a bit of
    // numerical fudge.
    bool no_remainder = contains_end[num_fine_time_points - 1]  &&
        membership_fraction[num_fine_time_points - 1] >= .9999;
    bool have_remainder = !no_remainder;
    num_coarse_time_points += have_remainder;

    BOOM::SubMatrix fine_series(REAL(r_fine_series),
                                num_fine_rows,
                                num_fine_time_points);
    BOOM::Mat coarse_series(num_fine_rows, num_coarse_time_points);

    for (int iteration = 0; iteration < num_fine_rows; ++iteration) {
      double current = 0;
      int coarse_time = 0;
      for (int fine_time = 0; fine_time < num_fine_time_points; ++fine_time) {
        if (contains_end[fine_time]) {
          current += fine_series(iteration, fine_time) *
              membership_fraction[fine_time];
          coarse_series(iteration, coarse_time) = current;
          ++coarse_time;
          current = (1 - membership_fraction[fine_time]) *
              fine_series(iteration, fine_time);
        } else {
          current += fine_series(iteration, fine_time);
        }
      }
      if (have_remainder) {
        coarse_series(iteration, coarse_time) = current;
      }
    }

    BOOM::RMemoryProtector protector;
    SEXP r_ans = protector.protect(
        Rf_isMatrix(r_fine_series)
        ? Rf_allocMatrix(REALSXP, num_fine_rows, num_coarse_time_points)
        : Rf_allocVector(REALSXP, num_coarse_time_points));
    double *ans = REAL(r_ans);
    std::copy(coarse_series.begin(), coarse_series.end(), ans);
    return r_ans;
  }
}

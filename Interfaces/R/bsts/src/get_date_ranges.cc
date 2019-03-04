/*
  Copyright (C) 2018 Steven L. Scott

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


#include <ctime>
#include <iostream>

#include "create_state_model.h"

#include "r_interface/boom_r_tools.hpp"
#include "r_interface/handle_exception.hpp"
#include "r_interface/list_io.hpp"
#include "r_interface/print_R_timestamp.hpp"
#include "r_interface/seed_rng_from_R.hpp"

#include "Models/StateSpace/StateModels/Holiday.hpp"
#include "cpputil/report_error.hpp"

extern "C" {
  using namespace BOOM;
  using std::endl;

  // Given a holiday and a sorted (increasing) vector of of timestamps, return a
  // two column matrix of timestamps giving the date ranges over which the
  // holiday is active.
  //
  // Args:
  //   r_holiday:  An R object inheriting from "Holiday".
  //   r_timestamps:  A vector containing an increasing sequence of Date objects.
  //
  // Returns:
  //   A two column matrix of integers giving the indices of the first and last
  //   active timepoints of each holiday period contained in timestamps.  The
  //   indices are unit-offset, so they're ready for use by R without adding 1
  //   to them.
  SEXP analysis_common_r_get_date_ranges_(
      SEXP r_holiday,
      SEXP r_timestamps) {
    try {
      Ptr<Holiday> holiday =
          BOOM::bsts::StateModelFactory::CreateHoliday(r_holiday);
      std::vector<Date> dates = BOOM::ToBoomDateVector(r_timestamps);
      std::vector<std::pair<int, int>> date_ranges;
      bool previous_day_was_holiday = false;
      int start = -1;
      int end = -1;
      for (int i = 0; i < dates.size(); ++i) {
        if (holiday->active(dates[i])) {
          if (!previous_day_was_holiday) {
            // Found the start of a new holiday. Add one to correct for R's
            // unit-offset counting scheme.
            start = i + 1;
          }
          previous_day_was_holiday = true;
        } else {
          if (previous_day_was_holiday) {
            // Found the end of a holiday.  Don't add 1 here because the end was
            // found to be the previous time point.
            end = i;
            date_ranges.push_back(std::make_pair(start, end));
            start = -1;
            end = -1;
            previous_day_was_holiday = false;
          }
        }
      }
      if (start > 0 && end < 0) {
        date_ranges.push_back(std::make_pair(start, dates.size()));
      }

      Matrix date_range_matrix(date_ranges.size(), 2);
      for (int i = 0; i < nrow(date_range_matrix); ++i) {
        date_range_matrix(i, 0) = date_ranges[i].first;
        date_range_matrix(i, 1) = date_ranges[i].second;
      }
      return ToRMatrix(date_range_matrix);
    } catch (std::exception &e) {
      RInterface::handle_exception(e);
    } catch(...) {
      RInterface::handle_unknown_exception();
    }
    return R_NilValue;
  }

}  // extern "C"

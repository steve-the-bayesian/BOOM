// Copyright 2020 Steven L. Scott. All Rights Reserved.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

#include "timestamp_info.h"
#include "r_interface/boom_r_tools.hpp"

namespace BOOM {
  namespace bsts {

    TimestampInfo::TimestampInfo(SEXP r_data_list) {
      Unpack(r_data_list);
    }

    void TimestampInfo::Unpack(SEXP r_data_list) {
      SEXP r_timestamp_info = getListElement(r_data_list, "timestamp.info");
      trivial_ = Rf_asLogical(getListElement(
          r_timestamp_info, "timestamps.are.trivial"));
      number_of_time_points_ = Rf_asInteger(getListElement(
          r_timestamp_info, "number.of.time.points"));
      if (!trivial_) {
        timestamp_mapping_ = ToIntVector(getListElement(
            r_timestamp_info, "timestamp.mapping"));
      }
    }

    // Args:
    //   r_prediction_data: A list containing an object named 'timestamps',
    //     which is a list containing the following objects.
    //     - timestamp.mapping: A vector of integers indicating the timestamp to
    //         which each observation belongs.
    //
    // Effects:
    //   The forecast_timestamps_ element in the TimestampInfo object gets
    //   populated.
    void TimestampInfo::UnpackForecastTimestamps(SEXP r_prediction_data) {
      SEXP r_forecast_timestamps = getListElement(
          r_prediction_data, "timestamps");
      if (!Rf_isNull(r_forecast_timestamps)) {
        forecast_timestamps_ = ToIntVector(getListElement(
            r_forecast_timestamps, "timestamp.mapping"));
        for (int i = 1; i < forecast_timestamps_.size(); ++i) {
          if (forecast_timestamps_[i] < forecast_timestamps_[i - 1]) {
            report_error("Time stamps for multiplex predictions must be "
                         "in increasing order.");
          }
        }
      }
    }
  }  // namespace bsts
}  // namespace BOOM

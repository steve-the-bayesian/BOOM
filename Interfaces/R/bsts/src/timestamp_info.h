#ifndef BSTS_SRC_TIMESTAMP_INFO_H_
#define BSTS_SRC_TIMESTAMP_INFO_H_
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

#include "r_interface/boom_r_tools.hpp"

namespace BOOM {
  namespace bsts {

    // A summary describing the timestamps accompanying the time series.
    class TimestampInfo {
     public:

      // Create a default TimestampInfo object.
      TimestampInfo() : trivial_(true),
                        number_of_time_points_(-1)
      {}

      // Create a C++ TimestampInfo object from an R TimestampInfo object.
      explicit TimestampInfo(SEXP r_data_list);

      // Args:
      //   r_data_list: A list containing an object named 'timestamp.info' which
      //     is an R object of class TimestampInfo.  The list contains the
      //     following named elements:
      //     - timestamps.are.trivial: Scalar boolean.
      //     - number.of.time.points:  Scalar integer.
      //     - timestamp.mapping: Either R_NilValue (if timestamps are trivial)
      //         or a numeric vector containing the index of the timestamp to
      //         which each observation belongs.  These indices are in R's
      //         unit-offset counting system.  The member function 'mapping'
      //         handles the conversion to the C++ 0-offset counting system.
      //
      // Effects:
      //   The timestamp.info object is extracted, and its contents are used to
      //   populate this object.
      void Unpack(SEXP r_data_list);

      void UnpackForecastTimestamps(SEXP r_prediction_data);

      void set_time_dimension(int dim) {
        number_of_time_points_ = dim;
      }

      bool trivial() const {return trivial_;}
      int number_of_time_points() const {return number_of_time_points_;}

      // The index of the time point to which observation i belongs.  The index
      // is in C's 0-based counting system.
      //
      // Args:
      //   observation_number: The index of an observation (row in the data)
      //     in C's 0-offset counting system.
      //
      // Returns:
      //   The index of the time point (again, in C's 0-offset counting system)
      //   to which the specified observation belongs.
      int mapping(int observation_number) const {
        return trivial_ ? observation_number
            : timestamp_mapping_[observation_number] - 1;
      }

      const std::vector<int> &forecast_timestamps() const {
        return forecast_timestamps_;
      }

     private:
      // Timestamps are trivial if the time points are uniformly spaced, no time
      // point is skipped, and there is a single observation per time point.
      bool trivial_;

      // The number of distinct time points.  Some of these might contain only
      // missing data.
      int number_of_time_points_;

      // timestamp_mapping_[i] gives the index of the time point to which
      // observation i belongs.  The indices are stored relative to 1 (as is the
      // custom in R).
      std::vector<int> timestamp_mapping_;

      // Indicates the number of time points past the end of the training data
      // for each forecast data point.  For example, if the next three time
      // points are to be forecast, this will be [1, 2, 3]. If data are not
      // multiplexed then forecast_timestamps_ will be empty.
      std::vector<int> forecast_timestamps_;
    };
  }  // namespace bsts
}  // namespace BOOM

#endif  // BSTS_SRC_TIMESTAMP_INFO_H_

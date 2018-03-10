// Copyright 2018 Google LLC. All Rights Reserved.
#ifndef BOOM_STATE_SPACE_MULTIPLEXED_DATA_HPP_
#define BOOM_STATE_SPACE_MULTIPLEXED_DATA_HPP_
/*
  Copyright (C) 2005-2017 Steven L. Scott

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

#include "Models/DataTypes.hpp"
#include "cpputil/Ptr.hpp"

namespace BOOM {
  namespace StateSpace {
    class MultiplexedData : public Data {
     public:
      MultiplexedData();

      // The observed sample size is the number of fully observed data points at
      // the time period described by this object.
      int observed_sample_size() const { return observed_sample_size_; }

      // The total_sample_size is the number of observed and missing data points
      // at the time period described by this object.
      virtual int total_sample_size() const = 0;

     protected:
      // Adjusts the missing status and observation count of the aggregate
      // multiplexed data object to reflect the missing status of dp.
      //
      // Child classes should call this function to update their missing-data
      // status and observation count in light of the new observation, but
      // actually storing the data is left to the class descendants.
      void add_data(const Ptr<Data> &dp);

     private:
      int observed_sample_size_;
    };
  }  // namespace StateSpace
}  // namespace BOOM

#endif  // BOOM_STATE_SPACE_MULTIPLEXED_DATA_HPP_

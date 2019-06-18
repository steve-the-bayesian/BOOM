// Copyright 2019 Steven L. Scott.
// Copyright 2018 Google LLC. All Rights Reserved.
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

#include "Models/StateSpace/MultiplexedData.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace StateSpace {
    MultiplexedData::MultiplexedData() : observed_sample_size_(0) {}

    // Child classes should call this function to make sure their missing status
    // and observed_sample_size_ are set correctly, but it does not actually
    // store data.
    void MultiplexedData::add_data(const Ptr<Data> &dp) {
      if (!dp) {
        report_error("A null data point wa passed to MultiplexedData::add_data.");
      }
      if (dp->missing() == Data::observed) {
        ++observed_sample_size_;
        if (this->missing() == Data::completely_missing) {
          set_missing_status(Data::partly_missing);
        }
      } else if (this->missing() == Data::observed) {
        if (observed_sample_size_ == 0) {
          set_missing_status(Data::completely_missing);
        } else {
          set_missing_status(Data::partly_missing);
        }
      }
    }

  }  // namespace StateSpace

}  // namespace BOOM

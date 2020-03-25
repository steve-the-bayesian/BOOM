/*
  Copyright (C) 2005-2020 Steven L. Scott

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

#include "Models/StateSpace/DynamicRegression.hpp"

namespace BOOM {
  namespace StateSpace {

    void RegressionDataTimePoint::add_data(const Ptr<RegressionData &dp) {
      if (suf_) {
        suf_->add_data(dp);
      } else {
        raw_data_.push_back(dp);
        if (raw_data_.size() >= dp->xdim()) {
          suf_.reset(new NeRegSuf(dp->xdim()));
          for (const &el : raw_data_) {
            suf_->add_data(el);
          }
          raw_data_.clear();
        }
      }
    }

  }  // namespace StateSpace


  void TimeSeriesRegressionDataPolicy::add_data(const Ptr<Data> &dp) {
    Ptr<RegressionData> reg_ptr = dp.dcast<RegressionData>();
    if (!!reg_ptr) {
      add_data(reg_ptr);
      return;
    }

    Ptr<StateSpace::RegressionDataTimePoint> time_point_ptr =
        dp.dcast<StateSpace::RegressionDataTimePoint>();
    if (!!time_point_ptr) {
      add_data(time_point_ptr);
    }

  }

}  // namespace BOOM

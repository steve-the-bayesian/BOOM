// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "stats/ECDF.hpp"
#include <algorithm>
#include <cmath>
#include "cpputil/report_error.hpp"

namespace BOOM {

  ECDF::ECDF(const ConstVectorView &unsorted_data)
      : sorted_data_(unsorted_data) {
    if (sorted_data_.empty()) {
      report_error("ECDF cannot be built from empty vector.");
    }
    // Remove any NaN's from the data.
    sorted_data_.erase(
        std::remove_if(sorted_data_.begin(), sorted_data_.end(),
                       [](double x){return std::isnan(x);}),
        sorted_data_.end());
    if (sorted_data_.empty()) {
      report_error("ECDF passed all NaN's.");
    }
    std::sort(sorted_data_.begin(), sorted_data_.end());
  }

  double ECDF::fplus(double x) const {
    if (sorted_data_.empty()) {
      report_error("An empty empirical CDF cannot be evaluated.");
    }
    double ans = std::upper_bound(sorted_data_.begin(), sorted_data_.end(), x) -
        sorted_data_.begin();
    return ans / sorted_data_.size();
  }

  double ECDF::fminus(double x) const {
    if (sorted_data_.empty()) {
      report_error("An empty empirical CDF cannot be evaluated.");
    }
    double ans = std::lower_bound(sorted_data_.begin(), sorted_data_.end(), x) -
        sorted_data_.begin();
    return ans / sorted_data_.size();
  }

  double ECDF::quantile(double probability) const {
    if (sorted_data_.empty()) {
      report_error("An empty empirical CDF cannot be evaluated.");
    }
    double min_prob = 1.0 / sorted_data_.size();

    if (probability < min_prob) {
      return sorted_data_[0];
    } else if (probability >= 1.0) {
      return sorted_data_.back();
    }

    int index = probability * sorted_data_.size();
    if ((sorted_data_.size() - probability * index) < min_prob
        || index + 1 == sorted_data_.size()) {
      return sorted_data_[index];
    } else {
      return .5 * (sorted_data_[index] + sorted_data_[index + 1]);
    }
  }

}  // namespace BOOM

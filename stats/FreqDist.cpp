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
#include "stats/FreqDist.hpp"
#include <iomanip>
#include <sstream>
#include "Models/CategoricalData.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  namespace {

    template <class INT>
    std::vector<int> count_values(const std::vector<INT> &y,
                                  std::vector<std::string> &labels,
                                  bool contiguous) {
      std::vector<int> counts;
      labels.clear();
      if (y.empty()) {
        return counts;
      }
      std::map<INT, int> distribution;
      for (int i = 0; i < y.size(); ++i) {
        ++distribution[y[i]];
      }
      if (contiguous) {
        INT smallest = distribution.begin()->first;
        INT largest = distribution.rbegin()->first;
        for (INT i = smallest; i <= largest; ++i) {
          // A hack to ensure that distribution[i] is accessed, and
          // thus populated (with zero, if it had not been populated
          // before).  No element will ever be less than zero.
          if (distribution[i] < 0) {
            distribution[i] = 0;
          }
        }
      }
      for (const auto &element : distribution) {
        counts.push_back(element.second);
        labels.push_back(std::to_string((element.first)));
      }
      return counts;
    }
  }  // namespace

  FrequencyDistribution::FrequencyDistribution(const std::vector<uint> &y,
                                               bool contiguous) {
    counts_ = count_values(y, labels_, contiguous);
  }

  FrequencyDistribution::FrequencyDistribution(const std::vector<int> &y,
                                               bool contiguous) {
    counts_ = count_values(y, labels_, contiguous);
  }

  FrequencyDistribution::FrequencyDistribution(
      const std::vector<unsigned long> &y, bool contiguous) {
    counts_ = count_values(y, labels_, contiguous);
  }

  FrequencyDistribution::FrequencyDistribution(
      const std::vector<int> &y, int lo, int hi) {
    counts_ = std::vector<int>(hi - lo + 1, 0);
    for (int yi : y) {
      ++counts_[yi - lo];
    }
    for (int i = lo; i <= hi; ++i) {
      labels_.push_back(std::to_string(i));
    }
  }

  void FrequencyDistribution::set_labels(
      const std::vector<std::string> &labels) {
  }

  std::ostream &FrequencyDistribution::print(std::ostream &out) const {
    uint N = labels_.size();
    uint labfw = 0;
    uint countfw = 0;
    for (uint i = 0; i < N; ++i) {
      uint len = labels_[i].size();
      if (len > labfw) labfw = len;

      std::string s = std::to_string(counts_[i]);
      len = s.size();
      if (len > countfw) countfw = len;
    }
    labfw += 2;
    countfw += 2;

    for (uint i = 0; i < N; ++i) {
      out << std::setw(labfw) << labels_[i] << std::setw(countfw)
          << counts_[i] << std::endl;
    }
    return out;
  }

  std::string FrequencyDistribution::mode() const {
    int max_count = -1;
    int mode_position = -1;
    for (int i = 0; i < counts_.size(); ++i) {
      if (counts_[i] > max_count) {
        max_count = counts_[i];
        mode_position = i;
      }
    }
    return labels_[mode_position];
  }

  int FrequencyDistribution::count(const std::string &label) const {
    for (int i = 0; i < labels_.size(); ++i) {
      if (labels_[i] == label) {
        return counts_[i];
      }
    }
    return 0;
  }

  void FrequencyDistribution::reset(const std::vector<int> &counts,
                                    const std::vector<std::string> &labels) {
    if (counts.size() != labels.size()) {
      report_error(
          "counts and labels must be the same size in "
          "FrequencyDistribution::reset");
    }
    counts_ = counts;
    labels_ = labels;
  }

  void FrequencyDistribution::set_default_labels() {
    labels_.clear();
    for (int i = 0; i < counts_.size(); ++i) {
      labels_.push_back(std::string("L") + std::to_string(i));
    }
  }

  BucketedFrequencyDistribution::BucketedFrequencyDistribution(
      const Vector &x, const Vector &cutpoints)
      : cutpoints_(sort(cutpoints)) {
    std::vector<int> counts(cutpoints.size() + 1, 0);
    Vector sorted_x = sort(x);
    int i = 0;
    for (int bucket = 0; bucket < cutpoints_.size(); ++bucket) {
      for (; i < x.size(); ++i) {
        if (sorted_x[i] > cutpoints_[bucket]) {
          break;
        }
        ++counts[bucket];
      }
    }
    counts.back() = sorted_x.size() - i;
    FrequencyDistribution::reset(counts, create_labels());
  }

  std::vector<std::string> BucketedFrequencyDistribution::create_labels()
      const {
    std::vector<std::string> ans;
    for (int i = 0; i < cutpoints_.size(); ++i) {
      std::ostringstream label_stream;
      if (i == 0) {
        label_stream << "(-inf";
      } else {
        label_stream << "(" << std::setprecision(3) << cutpoints_[i - 1];
      }
      label_stream << ", " << std::setprecision(3) << cutpoints_[i] << "]";
      ans.push_back(label_stream.str());
    }
    std::ostringstream label_stream;
    label_stream << "(" << std::setprecision(3) << cutpoints_.back()
                 << ", inf)";
    ans.push_back(label_stream.str());
    return ans;
  }
}  // namespace BOOM

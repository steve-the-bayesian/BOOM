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
#ifndef BOOM_FREQ_DIST_HPP
#define BOOM_FREQ_DIST_HPP

#include <string>
#include <vector>
#include "LinAlg/Vector.hpp"
#include "uint.hpp"

namespace BOOM {

  // A frequency distribution for categorical data.
  class FrequencyDistribution {
   public:
    // Overloaded constructors for various integral types.
    // Args:
    //   y:  A vector of categorical data to be tabulated.
    //   contiguous: If true, then all the values between the smallest and
    //     largest values in y will be included (with zero counts if they did
    //     not appear in y).  If false, then values that did not appear in y
    //     will be skipped.
    explicit FrequencyDistribution(const std::vector<uint> &y, bool contiguous = false);
    explicit FrequencyDistribution(const std::vector<int> &y, bool contiguous = false);
    explicit FrequencyDistribution(const std::vector<unsigned long> &y,
                                   bool contiguous = false);

    // Set the category labels for the unique values in y.
    void set_labels(const std::vector<std::string> &labels);
    const std::vector<std::string> &labels() const { return labs_; }

    // Count the frequency of each value in y.
    const std::vector<int> &counts() const { return counts_; }

    Vector relative_frequencies() const {
      Vector ans(counts_);
      double normalizing_constant = sum(ans);
      if (normalizing_constant == 0.0) {
        return ans;
      } else {
        return ans / normalizing_constant;
      }
    }

    std::ostream &print(std::ostream &out) const;

    // Returns the label corresponding to the largest count.  If two or more
    // levels tie then the first mode is reported.
    std::string mode() const;

    // Returns the count corresponding to a particular label.  Implemented using
    // linear search, so use this with care if the number of distinct labels is
    // large.
    //
    // Returns 0 if the label was not found.
    int count(const std::string &label) const;

   protected:
    FrequencyDistribution() {}
    void reset(const std::vector<int> &counts,
               const std::vector<std::string> &labels);

   private:
    std::vector<std::string> labs_;
    std::vector<int> counts_;
  };

  class BucketedFrequencyDistribution : public FrequencyDistribution {
   public:
    // Args:
    //   x:  A vector of real valued observations.
    //   cutpoints:  A vector of distinct cutpoint values.
    //
    // The constructor will sort the cutpoints, then count how many
    // times each cutpoints[i] < x <= cutpoints[i+1].
    BucketedFrequencyDistribution(const Vector &x, const Vector &cutpoints);

    // The vector of counts is one larger than the vector of cutpoints.
    const Vector &cutpoints() const { return cutpoints_; }

   private:
    Vector cutpoints_;

    std::vector<std::string> create_labels() const;
  };

  inline std::ostream &operator<<(std::ostream &out,
                                  const FrequencyDistribution &f) {
    return f.print(out);
  }

}  // namespace BOOM
#endif  // BOOM_FREQ_DIST_HPP

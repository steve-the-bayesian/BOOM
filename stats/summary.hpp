#ifndef BOOM_STATS_SUMMARY_HPP_
#define BOOM_STATS_SUMMARY_HPP_

/*
  Copyright (C) 2005-2018 Steven L. Scott

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

#include "LinAlg/VectorView.hpp"
#include <string>

namespace BOOM {

  // A collection of quantiles, plus the mean, for a numeric vector of data.
  class NumericSummary {
   public:
    explicit NumericSummary(const ConstVectorView &data);
    std::string to_string() const;
    std::ostream &print(std::ostream &out) const;

   private:
    double min_;
    double lower_quartile_;
    double median_;
    double mean_;
    double upper_quartile_;
    double max_;
  };

  std::ostream &operator<<(std::ostream &out, const NumericSummary &summary) {
    return summary.print(out);
  }
  
} // namespace BOOM

#endif //  BOOM_STATS_SUMMARY_HPP_


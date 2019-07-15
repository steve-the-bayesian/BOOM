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

#include "stats/summary.hpp"
#include "stats/ECDF.hpp"
#include "stats/moments.hpp"

#include <iomanip>
#include <sstream>

namespace BOOM {

  NumericSummary::NumericSummary(const ConstVectorView &data) {
    ECDF ecdf(data);
    min_ = ecdf.sorted_data()[0];
    lower_quartile_ = ecdf.quantile(.25);
    median_ = ecdf.quantile(.5);
    mean_ = mean(data);
    upper_quartile_ = ecdf.quantile(.75);
    max_ = ecdf.sorted_data().back();
  }

  std::string NumericSummary::to_string() const {
    std::ostringstream out;
    print(out);
    return out.str();
  }

  std::ostream &NumericSummary::print(std::ostream &out) const {
    using std::endl;
    auto precision = out.precision();
    out <<  "min:            " << std::setprecision(4) << min_ << endl
        <<  "lower quartile: " << lower_quartile_ << endl
        <<  "median:         " << median_ << endl
        <<  "mean:           " << mean_ << endl
        <<  "upper quartile: " << upper_quartile_ << endl
        <<  "max:            " << max_ << endl;
    out << std::setprecision(precision);
    return out;
  }

} // namespace BOOM

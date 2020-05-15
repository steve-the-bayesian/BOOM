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

//#include <iostream>

#include "test_utils/test_utils.hpp"
#include "stats/ECDF.hpp"
#include "stats/ks_critical_value.hpp"

namespace BOOM {

  bool DistributionsMatch(
      const Vector &data,
      const std::function<double(double)> &cdf,
      double significance) {

    ECDF ecdf(data);
    const Vector &sorted_data(ecdf.sorted_data());
    double maxdiff = negative_infinity();
    //    int64_t maxdiff_index = -1;
    //    double maxdiff_ecdf = negative_infinity();
    //double maxdiff_cdf = negative_infinity();
    for (int64_t i = 0; i < sorted_data.size(); ++i) {
      double delta = fabs(ecdf(sorted_data[i]) - cdf(sorted_data[i]));
      if (delta > maxdiff) {
        maxdiff = delta;
        //   maxdiff_index = i;
        //   maxdiff_ecdf = ecdf(sorted_data[i]);
        //   maxdiff_cdf = cdf(sorted_data[i]);
      }
    }

    double critical_value = ks_critical_value(sorted_data.size(), significance);
    // if (maxdiff > critical_value) {
    //   std::cout << "maximum difference of " << maxdiff
    //             << " occurred at position " << maxdiff_index
    //             << " with ECDF = " << maxdiff_ecdf << " and CDF = "
    //             << maxdiff_cdf << std::endl;
    // }

    return maxdiff <= critical_value;
  }

  bool TwoSampleKs(
      const ConstVectorView &data1,
      const ConstVectorView &data2,
      double significance) {
    ECDF ecdf1(data1);
    ECDF ecdf2(data2);
    double maxdiff = negative_infinity();
    for (double x : ecdf1.sorted_data()) {
      maxdiff = std::max<double>(maxdiff, fabs(ecdf1(x) - ecdf2(x)));
    }
    for (double x : ecdf2.sorted_data()) {
      maxdiff = std::max<double>(maxdiff, fabs(ecdf1(x) - ecdf2(x)));
    }
    double n1 = data1.size();
    double n2 = data2.size();
    double constant = sqrt(-.5 * log(significance / 2));

    double critical_value = constant * sqrt((n1 + n2) / (n1 * n2));
    return maxdiff <= critical_value;
  }

  namespace {
    bool CheckCoverage(const ECDF &ecdf1, const ECDF &ecdf2) {
      // Compute the (.2, .8) central interval for ECDF1.
      double lo1 = ecdf1.quantile(.2);
      double hi1 = ecdf1.quantile(.8);

      // Compute the (.3, .7) interval for ECDF2.
      double lo2 = ecdf2.quantile(.3);
      double hi2 = ecdf2.quantile(.7);

      // If the distributions are nearly the same, interval1 should cover
      // interval2.
      return lo1 <= lo2 && hi1 >= hi2;
    }
  }  // namespace

  bool EquivalentSimulations(const ConstVectorView &draws1,
                             const ConstVectorView &draws2) {
    ECDF ecdf1(draws1);
    ECDF ecdf2(draws2);
    return CheckCoverage(ecdf1, ecdf2) && CheckCoverage(ecdf2, ecdf1);
  }

}  // namespace BOOM

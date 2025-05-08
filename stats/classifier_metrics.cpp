/*
  Copyright (C) 2005-2025 Steven L. Scott

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

#include "stats/classifier_metrics.hpp"
#include "uint.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  double TjurR2(const std::vector<bool> &truth,
                const Vector &predicted_probabilities) {

    double success_prob_sum = 0.0;
    double failure_prob_sum = 0.0;
    Int success_count = 0;
    Int failure_count = 0;

    if (truth.size() != predicted_probabilities.size()) {
      std::ostringstream err;
      err << "The truth vector ("
          << truth.size()
          << ") and the vector of predicted probabilities ("
          << predicted_probabilities.size()
          << ") must have the same size.";
      report_error(err.str());
    }

    for (Int i = 0; i < truth.size(); ++i) {
      if (truth[i]) {
        ++success_count;
        success_prob_sum += predicted_probabilities[i];
      } else {
        ++failure_count;
        failure_prob_sum += predicted_probabilities[i];
      }
    }

    if (success_count == 0 || failure_count == 0) {
      report_error("There must be ast least one success and one "
                   "failure to compute Tjur R2.");
    }

    double ans = (success_prob_sum / success_count)
                 - (failure_prob_sum / failure_count);
    if (ans > 1.0) {
      ans = 1.0;
    } else if (ans < 0.0) {
      ans = 0.0;
    }

    return ans;
    
  }

}  // namespace BOOM

/*
  Copyright (C) 2005-2023 Steven L. Scott

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

#include "test_utils/test_utils.hpp"
#include "cpputil/report_error.hpp"
#include "stats/moments.hpp"
#include <fstream>

namespace BOOM {
  std::string CheckStochasticProcess(const Matrix &draws,
                                     const Vector &truth,
                                     double confidence,
                                     double sd_ratio_threshold,
                                     double coverage_fraction,
                                     const std::string &filename) {
    ostringstream err;
    Matrix centered_draws = draws;
    double number_covering = 0;
    for (int i = 0; i < ncol(centered_draws); ++i) {
      centered_draws.col(i) -= truth[i];
      number_covering += covers(draws.col(i), truth[i], confidence);
    }
    number_covering /=  ncol(draws);
    if (number_covering < coverage_fraction) {
      err << "fewer than half the intervals covered the true value.  "
          << "Coverage fraction = " << number_covering << "."
          << std::endl;
    }

    Vector means = mean(centered_draws);
    double truth_sd = sd(truth);
    double residual_sd = sd(means);

    if (residual_sd / truth_sd > sd_ratio_threshold) {
      err << "The standard deviation of the centered draws (centered "
          << "around true values) is " << residual_sd << ". \n"
          << "The standard deviation of the true function is "
          << truth_sd << ".\n"
          << "The ratio is " << residual_sd / truth_sd
          << " which exceeds the testing threshold of "
          << sd_ratio_threshold << "." << std::endl;
    }

    std::string ans = err.str();
    if (ans != "") {
      std::ofstream error_file(filename);
      error_file << truth << std::endl << draws;
    }
    return ans;
  }

}  // namespace BOOM

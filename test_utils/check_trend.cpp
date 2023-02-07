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
#include "stats/quantile.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include <fstream>

namespace BOOM {

  bool CheckTrend(const Matrix &draws, const Vector &truth, double r2_threshold) {
    if (draws.ncol() != truth.size()) {
      report_error("The number of columns in 'draws' must match the length "
                   "of 'truth'.");
    }
    Vector trend = median(draws);
    Matrix X = cbind(Vector(truth.size(), 1.0), truth);
    RegressionModel model(X, trend);
    return model.anova().Rsquare() >= r2_threshold;
  }

}  // namespace BOOM

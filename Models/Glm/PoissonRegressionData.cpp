// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2013 Steven L. Scott

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

#include "Models/Glm/PoissonRegressionData.hpp"
#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  PoissonRegressionData::PoissonRegressionData(int64_t y, const Vector &x,
                                               double exposure)
      : PoissonRegressionData(y, new VectorData(x), exposure) {}

  PoissonRegressionData::PoissonRegressionData(int64_t y,
                                               const Ptr<VectorData> &x,
                                               double exposure)
      : GlmData<IntData>(Ptr<IntData>(new IntData(y)), x),
        exposure_(exposure),
        log_exposure_(log(exposure)) {
    if (y < 0) {
      report_error(
          "Negative value of 'y' passed to "
          "PoissonRegressionData constructor.");
    }
    if (exposure < 0) {
      report_error(
          "You can't pass a negative exposure to the "
          "PoissonRegressionData constructor.");
    }
    if (exposure == 0 && y > 0) {
      report_error(
          "If exposure is 0 then y must also be 0 in "
          "PoissonRegressionData constructor.");
    }
  }

  PoissonRegressionData *PoissonRegressionData::clone() const {
    return new PoissonRegressionData(*this);
  }

  std::ostream &PoissonRegressionData::display(std::ostream &out) const {
    out << "[" << exposure_ << "]  ";
    return GlmData<IntData>::display(out);
  }

  double PoissonRegressionData::exposure() const { return exposure_; }

  double PoissonRegressionData::log_exposure() const { return log_exposure_; }

  void PoissonRegressionData::set_exposure(double exposure, bool signal) {
    if (exposure < 0) {
      report_error("Exposure must be non-negative");
    } else if (exposure <= 0.0) {
      exposure_ = 0.0;
      log_exposure_ = negative_infinity();
    } else {
      exposure_ = exposure;
      log_exposure_ = log(exposure);
    }
    if (signal) {
      Data::signal();
    }
  }

}  // namespace BOOM

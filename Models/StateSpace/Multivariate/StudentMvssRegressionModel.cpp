/*
  Copyright (C) 2005-2022 Steven L. Scott

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

#include "Models/StateSpace/Multivariate/StudentMvRegModel.hpp"
namespace BOOM {
  namespace {
    using StudentData = StudentMultivariateTimeSeriesRegressionData;
  }

  StudentData::StudentMultivariateTimeSeriesRegressionData(
      double y, const Vector &x, int series, int timestamp)
      : MultivariateTimeSeriesRegressionData(y, x, series, timestamp),
        weight_(1.0)
  {}

  StudentData::StudentMultivariateTimeSeriesRegressionData(
      const Ptr<DoubleData> & y,
      const Ptr<VectorData> &x,
      int series,
      int timestamp)
      : MultivariateTimeSeriesRegressionData(y, x, series, timestamp),
        weight_(1.0)

}  // namespace BOOM

#ifndef BOOM_STATS_QUANTILES_HPP_
#define BOOM_STATS_QUANTILES_HPP_
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

#include "LinAlg/VectorView.hpp"

namespace BOOM {

  // Return a specific quantile of the input data.
  //
  // Args:
  //   data:  The data to be analyzed.
  //   target_quantile:  The quantile of the data to be returned.
  double quantile(const ConstVectorView &data,
                  double target_quantile);

  // Return a collection of quantiles from the input data.  This is more
  // efficient than calling 'quantile' on each of the target quantiles, because
  // the data only gets sorted once.
  //
  // Args:
  //   data:  The data to be analyzed.
  //   target_quantile:  The collection of quantiles to be returned.
  Vector quantile(const ConstVectorView &data,
                  const Vector &target_quantiles);

  // Return a specific quantile on each column of data.
  //
  // Args:
  //   data:  The data to be analyzed.
  //   target_quantile:  The quantile of the data to be returned.
  Vector quantile(const Matrix &data,
                  double target_quantile);

  // Return a collection of quantiles on each column of data.
  //
  // Args:
  //   data:  The data to be analyzed.
  //   target_quantile:  The collection of target quantiles to be returned.
  //
  // Returns: A Matrix with columns matching 'data' and rows corresponding to
  //   the target quantiles.
  Matrix quantile(const Matrix &data,
                  const Vector &target_quantiles);

  inline double median(const ConstVectorView &data) {
    return quantile(data, .5);
  }

  // Return the column-by-column median of 'data'.
  inline Vector median(const Matrix &data) {
    return quantile(data, 0.5);
  }

}

#endif  // BOOM_STATS_QUANTILES_HPP_

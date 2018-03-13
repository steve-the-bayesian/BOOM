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

#ifndef BOOM_COMPARE_PREDICTIONS_TEST_HPP_
#define BOOM_COMPARE_PREDICTIONS_TEST_HPP_
#include "LinAlg/VectorView.hpp"

namespace BOOM {

  // A hypothesis test that predicted values are randomly distributed
  // around true values.  The test imagines plotting predictions vs
  // truth, and testing the coefficients of the regression with the
  // null that beta0 = 0 && beta1 = 0.
  struct ComparePredictionsOutput {
    double intercept;
    double intercept_se;
    double slope;
    double slope_se;
    double SSE;
    double SST;
    double Fstat;
    double p_value;
  };

  std::ostream &operator<<(std::ostream &out,
                           const ComparePredictionsOutput &output);

  ComparePredictionsOutput compare_predictions(const Vector &truth,
                                               const Vector &predictions);
  ComparePredictionsOutput compare_predictions(
      const ConstVectorView &truth, const ConstVectorView &predictions);

}  // namespace BOOM
#endif  // BOOM_COMPARE_PREDICTIONS_TEST_HPP_

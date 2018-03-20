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

#include "stats/compare_predictions.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "Models/Glm/PosteriorSamplers/RegressionConjSampler.hpp"
#include "Models/Glm/RegressionModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  ostream &operator<<(ostream &out, const ComparePredictionsOutput &cmp) {
    out << " intercept:  " << cmp.intercept << endl
        << " (SE)     :  " << cmp.intercept_se << endl
        << " slope    :  " << cmp.slope << endl
        << " (SE)     :  " << cmp.slope_se << endl
        << " SSE      :  " << cmp.SSE << endl
        << " SST      :  " << cmp.SST << endl
        << " F        :  " << cmp.Fstat << endl
        << " p_value  :  " << cmp.p_value << endl;
    return out;
  }

  ComparePredictionsOutput compare_predictions(const ConstVectorView &truth,
                                               const ConstVectorView &pred) {
    Matrix X(truth.size(), 2);
    X.col(0) = 1.0;
    X.col(1) = truth;
    RegressionModel model(X, pred);
    Vector null_residual = pred - truth;
    Vector beta = model.Beta();
    Vector alternative_residual = pred - X * beta;
    // Under the null hypothesis, residual and alternative_residual
    // will have the same distribution, but the alternative_residual
    // will have used two degrees of freedom, while the null reisdual
    // will have used zero.
    int n = truth.size();
    double SSE = alternative_residual.normsq();
    double SST = null_residual.normsq();
    double SSR = SST - SSE;
    double Fstat = (SSE / (n - 2)) / (SSR / 2.0);
    double p_value = pf(Fstat, n - 2, 2, false);
    ComparePredictionsOutput result;

    SpdMatrix xtx(2, 0.0);
    xtx.add_inner(X);
    Vector beta_standard_errors = sqrt(model.sigsq() * (xtx.inv().diag()));

    result.intercept = beta[0];
    result.intercept_se = beta_standard_errors[0];
    result.slope = beta[1];
    result.slope_se = beta_standard_errors[1];
    result.SSE = SSE;
    result.SST = SST;
    result.Fstat = Fstat;
    result.p_value = p_value;
    return result;
  }

  ComparePredictionsOutput compare_predictions(const Vector &truth,
                                               const Vector &pred) {
    return compare_predictions(ConstVectorView(truth), ConstVectorView(pred));
  }

}  // namespace BOOM

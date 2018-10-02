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

#include "Models/MatrixNormalModel.hpp"
#include "distributions.hpp"

namespace BOOM {
  MatrixNormalModel::MatrixNormalModel(int nrow, int ncol)
      : ParamPolicy_3(new MatrixParams(Matrix(nrow, ncol, 0.0)),
                      new SpdParams(nrow),
                      new SpdParams(ncol))
  {}

  MatrixNormalModel::MatrixNormalModel(const Matrix &mu,
                                       const SpdMatrix &row_variance,
                                       const SpdMatrix &column_variance)
      : ParamPolicy_3(new MatrixParams(mu),
                      new SpdParams(row_variance),
                      new SpdParams(column_variance))
  {}

  const Vector &MatrixNormalModel::mu() const {
    mean_workspace_ = vec(mean());
    return mean_workspace_;
  }

  const SpdMatrix &MatrixNormalModel::Sigma() const {
    variance_workspace_ = Kronecker(column_variance(), row_variance());
    return variance_workspace_;
  }
  
  const SpdMatrix &MatrixNormalModel::siginv() const {
    variance_workspace_ = Kronecker(column_precision(), row_precision());
    return variance_workspace_;
  }

  Vector MatrixNormalModel::mvn_mean() const {
    return vec(mean());
  }

  SpdMatrix MatrixNormalModel::mvn_variance() const {
    return Kronecker(column_variance(), row_variance());
  }

  SpdMatrix MatrixNormalModel::mvn_precision() const {
    return Kronecker(column_precision(), row_precision());
  }

  double MatrixNormalModel::logp(const Matrix &y) const {
    return dmatrix_normal_ivar(y, mean(),
                               row_precision(), row_precision_logdet(),
                               column_precision(), column_precision_logdet(),
                               true);
  }

  double MatrixNormalModel::logp(const Vector &y) const {
    return logp(Matrix(nrow(), ncol(), y));
  }

  Matrix MatrixNormalModel::simulate(RNG &rng) const {
    Matrix Z(nrow(), ncol());
    for (int i = 0; i < nrow(); ++i) {
      for (int j = 0; j < ncol(); ++j) {
        Z(i, j) = rnorm_mt(rng);
      }
    }
    return mean() + (row_variance_param()->var_chol() * Z).multT(
        column_variance_param()->var_chol());
  }

  Vector MatrixNormalModel::sim(RNG &rng) const {
    return vec(simulate(rng));
  }
  
}  // namespace BOOM

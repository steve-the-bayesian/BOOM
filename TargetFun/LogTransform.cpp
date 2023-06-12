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

#include "TargetFun/LogTransform.hpp"

namespace BOOM {

  double LogTransformJacobian::logdet(const Vector &raw) const {
    return log(raw).sum();
  }

  Matrix LogTransformJacobian::matrix(const Vector &raw) const {
    Matrix ans(raw.size(), raw.size(), 0.0);
    ans.diag() = raw;
    return ans;
  }

  SpdMatrix LogTransformJacobian::inverse_matrix(const Vector &raw) const {
    Matrix ans(raw.size(), raw.size(), 0.0);
    ans.diag() = 1.0 / raw;
    return ans;
  }

  void LogTransformJacobian::add_logs_gradient(
      const Vector &raw, Vector &gradient, bool positive) const {
    if (positive) {
      gradient += 1.0;
    } else {
      gradient -= 1.0;
    }
  }

  void LogTransformJacobian::add_logdet_gradient(
      Vector &gradient, const Vector &raw) {
    gradient += 1;
  }


}  // namespace BOOM

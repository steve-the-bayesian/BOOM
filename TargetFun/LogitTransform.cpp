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

#include "TargetFun/LogitTransform.hpp"
#include "stats/logit.hpp"

namespace BOOM {

  double LogitTransformJacobian::logdet(const Vector &probs) const {
    double ans = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
      ans += log(probs[i]) + log(1 - probs[i]);
    }
    return ans;
  }

  Matrix LogitTransformJacobian::matrix(const Vector &probs) const {
    Matrix ans(probs.size(), probs.size(), 0.0);
    diag(ans) = probs * (1 - probs);
    return ans;
  }

  void LogitTransformJacobian::add_logits_gradient(
      const Vector &probs, Vector &gradient, bool positive) const {
    Vector probs_complement = 1.0 - probs;
    Vector dp = probs * probs_complement;
    double sign = positive ? 1.0 : -1.0;
    gradient += sign * (dp / probs - dp / probs_complement);
  }

  Vector LogitTransform::transform(const Vector &probs) {
    return logit(probs);
  }

  Vector LogitTransform::inverse_transform(const Vector &logits) {
    return logit_inv(logits);
  }


}  // namespace BOOM

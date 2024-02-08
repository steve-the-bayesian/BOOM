/*
  Copyright (C) 2005-2024 Steven L. Scott

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

#include "TargetFun/SumMultinomialLogitTransform.hpp"

namespace BOOM {

  Vector SumMultinomialLogitTransform::to_logits(
      const Vector &positive_numbers) const {
    Vector ans = log(positive_numbers / positive_numbers[0]);
    ans[0] = positive_numbers.sum();
    return ans;
  }

  Vector SumMultinomialLogitTransform::from_logits(
      const Vector &sum_and_logits) const {
    double total = sum_and_logits[0];
    ConstVectorView logits(sum_and_logits, 1);

    Vector ans = exp(sum_and_logits);
    ans[0] = 1.0;
    double normalizing_constant = sum(ans);
    ans *= (total / normalizing_constant);
    return ans;
  }



}  // namespace BOOM

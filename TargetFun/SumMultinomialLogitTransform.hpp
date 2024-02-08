#ifndef BOOM_TARGETFUN_SUM_MULTINOMIALLOGIT_TRANSFORM_HPP_
#define BOOM_TARGETFUN_SUM_MULTINOMIALLOGIT_TRANSFORM_HPP_

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

#include "Targetfun/Transformation.hpp"

namespace BOOM {

  // Transforms lambda_1, ..., lambda_N to a new vector whose first term is
  // lambda_1 + ... + lambda_N and whose subsequent terms are log(lambda_k /
  // lambda_1) for k = 2...N.
  class SumMultinomialLogitTransform {
   public:

    // The forward transform.
    //
    // Args:
    //   positive_numbers:
    // Returns:
    //   A Vector of the same dimension as positive_numbers.  The first element
    //   is the sum of the positive_numbers.  Remaining elements are the
    //   multinomial logit transform of the positive_numbers, with the first
    //   element as the reference category.
    Vector to_logits(const Vector &positive_numbers) const;


    // Performs the inverse transformation of to_logits.
    //
    //
    Vector from_logits(const Vector &sum_and_logits) const {
      ConstVectorView logits(sum_and_logits, 1);
      Vector exp_logits = exp(logits);
      double nc = 1 + sum(exp(logits));
      double total = sum_and_logits[0];
      Vector ans(sum_and_logits.size());
      ans[0] = total;
      VectorView(ans, 1) = exp_logits;
      ans /= nc;
      return ans;
    }

   private:
  };

  // The SumMultinomialLogitTransform is very closely related to the multinomial
  // logit transform, so the Jacobian unsurprisingly can be implemented in terms
  // of that transform.
  class SumMultinomialLogitJacobian
      : public Jacobian {
   public:

    double element(int r, int s, const Vector &positive_numbers) const;

    Matrix matrix(const Vector &positive_numbers) const override;

    double logdet(const Vector &positive_numbers) const override;

   private:
  };

}  // namespace BOOM


#endif  // BOOM_TARGETFUN_SUM_MULTINOMIALLOGIT_TRANSFORM_HPP_

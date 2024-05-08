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

#include "TargetFun/Transformation.hpp"

namespace BOOM {

  // Transforms lambda_1, ..., lambda_N to a new vector whose first term is
  // lambda_1 + ... + lambda_N and whose subsequent terms are log(lambda_k /
  // lambda_1) for k = 2...N.
  class SumMultinomialLogitTransform {
   public:

    // Args:
    //   positive_numbers:
    // Returns:
    //   A Vector of the same dimension as positive_numbers.  The first element
    //   is the sum of the positive_numbers.  Remaining elements are the
    //   multinomial logit transform of the positive_numbers, with the first
    //   element as the reference category.
    //
    // Example:  if positive_numbers = {1, 2, 3} then the forward transform is
    // {1 + 2 + 3, log(2/1), log(3/1)}.
    Vector to_sum_logits(const Vector &positive_numbers) const;

    // Performs the inverse transformation of to_logits.
    //
    // Args:
    //   sum_and_logits: A Vector of values that could have been produced by
    //     to_sum_logits.  The first entry must be positive.
    //
    // Returns:
    //   Let 'total = sum_and_logits[0]' and let eta[s] = sum_and_logits[s] for
    //   s > 0.  Let NC = 1 + sum(exp(eta[s])) where the sum is over s > 0.
    //   Then the returned value is
    //     {total / NC,
    //      total * exp(eta[1]) / NC,
    //      ...,
    //      total * exp(eta[N-1]) / NC}.
    Vector from_sum_logits(const Vector &sum_and_logits) const;
  };

  class SumMultinomialLogitJacobian : public Jacobian {
   public:
    // Return the r,s element of the Jacobian matrix.
    double element(int r, int s, const Vector &positive_numbers) const;

    // The derivative of positive_numbers with respect to sum_and_logits.
    Matrix matrix(const Vector &positive_numbers) const override;

    // The log of the absolute value of the determinant of the Jacobian matrix,
    // evaluated at positive_numbers.
    double logdet(const Vector &positive_numbers) const override;

    // Add the gradient of logdet (with respect to sum_and_logits), evaluated at
    // positive_numbers.
    void add_logdet_gradient(Vector &gradient,
                             const Vector &positive_numbers) override;

    void add_logdet_Hessian(Matrix &hessian,
                            const Vector &positive_numbers) override;

    // The derivative of J(s, t) with respect to eta[r].
    double second_order_element(
        int r, int s, int t, const Vector &positive_numbers) const override;

    // The inverse of the Jacobian matrix, evaluated at positive_numbers.
    Matrix inverse_matrix(const Vector &positive_numbers) const;

    // The log determinant of the inverse Jacobian matrix, evaluated at
    // positive_numbers.
    double logdet_inverse_matrix(const Vector &positive_numbers) const;
  };

}  // namespace BOOM


#endif  // BOOM_TARGETFUN_SUM_MULTINOMIALLOGIT_TRANSFORM_HPP_

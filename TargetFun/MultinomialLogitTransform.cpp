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

#include "TargetFun/MultinomialLogitTransform.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  Vector MultinomialLogitTransform::to_logits(const Vector &probs, bool truncated) const {
    if (truncated) {
      double p0 = 1.0 - probs.sum();
      return log(probs / p0);
    } else {
      return Vector(ConstVectorView(log(probs / probs[0]), 1));
    }
  }

  Vector MultinomialLogitTransform::to_probs(const Vector &logits, bool truncated_probs) const {
    Vector ans = to_probs_full(concat(0, logits));
    if (truncated_probs) {
      return ConstVectorView(ans, 1);
    } else {
      return ans;
    }
  }

  Vector MultinomialLogitTransform::to_probs_full(
      const Vector &logits_including_implict_zero) const {
    Vector ans = logits_including_implict_zero;
    ans.normalize_logprob();
    return ans;
  }

  // Here is the math for the Jacobian of the multinomial logit
  // transform of probs:  probs -> logits.
  //
  // The inverse transform is defined as
  // probs[s] = exp(logits[s]) / (1 + sum_r(exp(logits[r]))).
  //
  // Using the quotient rule, we have for the (r,s) element of the
  // Jacobian matrix:
  //     d_probs[s] / d_logits[r != s]
  //         = 0 - exp(logits[s]) * exp(logits[r]) / (1 + sum)^2
  //         = -probs[s] * probs[r]
  // and d_probs[s] / d_logits[s]
  //         = [(1 + sum)*exp(logits[s]) - exp(logits[s]) * exp(logits[s])]
  //             / (1 + sum)^2
  //         = probs[s] - probs[s]^2
  //
  // In matrix form this says d_probs / d_logits = diag(probs) - probs *
  // probs^T.  Keep in mind that the first element of probs is omitted,
  // so this is a square matrix with side model_->nu().size() - 1.
  //
  // Args:
  //   truncated_probs: The probability vector with the first element removed.
  //
  // Returns:
  //   The matrix d_probs / d_logits.  See above.
  Matrix MultinomialLogitJacobian::matrix(const Vector &truncated_probs) const {
    int dim = truncated_probs.size();
    Matrix ans(dim, dim, 0.0);
    ans.diag() = truncated_probs;
    ans.add_outer(truncated_probs, truncated_probs, -1.0);
    return ans;
  }

  // Args:
  //   truncated_probs: The probability vector with the first element removed.
  //
  // Returns:
  //   The log determinant of the Jacobian matrix produced by matrix().
  double MultinomialLogitJacobian::logdet(const Vector &truncated_probs) const {
    double probs0 = 1.0 - truncated_probs.sum();
    double ans = log(probs0);
    for (int s = 0; s < truncated_probs.size(); ++s) {
      ans += log(truncated_probs[s]);
    }
    return ans;
  }

  // Here is the math for the inverse Jacobian transformation.
  //
  // logits[s] = log(probs[s] / (1 - sum(probs))) = log(probs[s]) - log(probs[0])
  //  d logits[s] / d probs [r != s]
  //       = -(1.0 / probs0) * (-1)
  //       = 1.0 / probs0
  //  The extra (-1) in the second line comes from the chain rule,
  //  and the fact that probs[0] = 1 - sum(probs).
  //
  //  d logits[s] / d probs[s]
  //       = (1.0 / probs[s]) + (1.0 / probs0)
  //
  // The matrix way of expressing this is that the inverse Jacobian
  // is diag(1.0 / truncated_probs) + 1/probs0 * b1 * b1^T, where b1 is
  // a vector of 1's.
  //
  // Args:
  //   truncated_probs: The probability vector with the first element removed.
  //
  // Returns:
  //   The inverse of the Jacobian matrix produced by matrix().
  SpdMatrix MultinomialLogitJacobian::inverse_matrix(const Vector &truncated_probs) const {
    SpdMatrix ans(truncated_probs.size());
    double probs0 = 1.0 - sum(truncated_probs);
    ans = 1.0 / probs0;
    diag(ans) += (1.0 / truncated_probs);
    return ans;
  }

  // From Harville, 15.8.6:
  //     d logdet(J) / d logits[s] =   tr (J^{-1} * d_J/d_logits[s])
  //
  // tr(AB) = sum_i sum_j (a[i,j] * b[j, i])
  // Element i, j of dJ / d_logits[s] is this->second_order_element(s, i, j).
  //
  // But even simpler...
  // log |J| = sum_i log(full_probs[i])     (probs is truncated_probs)
  //     d_logdet(J) / d_logits[r]
  //      = (1/probs0) * (-1) * sum_s(J[r,s]) + sum_s (1/probs[s]) J(r,s)
  //      = sum_s J(r, s) * (1/probs[s] - 1/probs0)
  //
  // Args:
  //   truncated_probs: The probability vector with the first element removed.
  //   gradient: The gradient vector to be modified.
  //   jacobian_matrix:  The Jacobian matrix produced by matrix().
  //   positive: If true then the derivative of the Jacobian's log determinant
  //     is add to 'gradient'.  If false it is subtracted.
  //
  // Effects:
  //   The derivative of logdet(J) with respect to logits is added to
  //   'gradient'.
  void MultinomialLogitJacobian::add_logits_gradient(
      const Vector &truncated_probs,
      Vector &gradient,
      const SpdMatrix &jacobian_matrix,
      bool positive) const {
    if (gradient.size() != truncated_probs.size()) {
      report_error("gradient is the wrong size.");
    }
    double probs0 = 1.0 - sum(truncated_probs);
    Vector adjusted_reciprocal = 1.0 / truncated_probs;
    adjusted_reciprocal -= 1.0 / probs0;
    gradient += jacobian_matrix * adjusted_reciprocal;
    if (positive) {
      gradient += jacobian_matrix * adjusted_reciprocal;
    } else {
      gradient -= jacobian_matrix * adjusted_reciprocal;
    }
  }

  void MultinomialLogitJacobian::add_logdet_gradient(
      Vector &gradient, const Vector &truncated_probs) {
    add_logits_gradient(truncated_probs,
                        gradient,
                        matrix(truncated_probs),
                        true);
  }

  // From Harville 15.9.3:
  // Write Jinv = J^{-1}.
  //
  //   d^2 log|J| / d_logits[r] d_logits[s] =
  //     = tr(Jinv d^2J / d_logits[r] d_logits[s])
  //       - tr( Jinv*(d_J / d_logits[r]) * Jinv*(d_J / d_logits[s]) )
  // Where dJ/dlogits[r] can be obtained through a call to
  // second_order.
  //
  // Even simpler:
  // From the comment to add_logits_gradient we have
  //    d_logdet(J) / d_logits[s] = \sum_t J(s,t) * (1/probs[t] - 1/probs0)
  //        d . / d_logits[r]
  //          = \sum_t d_J(s,t) / d_logits[r] * (1/probs[t] - 1/probs0)
  //                +  J(s,t) * (-1/probs[t]^2 * J(r,t)
  //                             +1/probs0^2 * d_probs0 / d_logits[r])
  //          = \sum_t J2(r,s,t) * (1/probs[t] - 1/probs0)
  //                   - J(s,t) * J(r,t) / probs[t]^2
  //                   - J(s,t) * sum_m J(r,m) / probs0^2
  //
  // Args:
  //   truncated_probs: The probability vector with the first element removed.
  //   hessian: The Hessian matrix to be modified.
  //   jacobian_matrix:  The Jacobian matrix produced by matrix().
  //   positive: If true then the derivative of the Jacobian's log determinant
  //     is add to 'gradient'.  If false it is subtracted.
  //
  // Effects:
  //   The Hessian of logdet(J) with respect to logits is added to
  //   'hessian'.
  void MultinomialLogitJacobian::add_logits_hessian(
      const Vector &truncated_probs,
      Matrix &hessian,
      const SpdMatrix &jacobian_matrix,
      bool positive) const {
    double sign = positive ? 1.0 : -1.0;
    const SpdMatrix &J(jacobian_matrix);
    int dim = hessian.nrow();
    double probs0 = 1.0 - sum(truncated_probs);
    for (int r = 0; r < dim; ++r) {
      double jacobian_row_sum_over_probs0_squared =
          sum(J.row(r)) / square(probs0);
      for (int s = 0; s < dim; ++s) {
        for (int t = 0; t < dim; ++t) {
          hessian(r, s) +=
              sign * second_order_element(r, s, t, truncated_probs) *
              (1.0 / truncated_probs[t] - 1.0 / probs0) -
              J(s, t) * (J(r, t) / square(truncated_probs[t]) +
                         jacobian_row_sum_over_probs0_squared);
        }
      }
    }
  }

  void MultinomialLogitJacobian::add_logdet_Hessian(
      Matrix &hessian,
      const Vector &truncated_probs) {
    add_logits_hessian(truncated_probs, hessian, matrix(truncated_probs), true);
  }

}  // namespace BOOM

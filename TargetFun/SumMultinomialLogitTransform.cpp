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
#include "TargetFun/MultinomialLogitTransform.hpp"
#include "cpputil/report_error.hpp"
#include <cmath>
#include "cpputil/math_utils.hpp"

namespace BOOM {

  Vector SumMultinomialLogitTransform::to_sum_logits(
      const Vector &positive_numbers) const {
    Vector ans = log(positive_numbers / positive_numbers[0]);
    ans[0] = positive_numbers.sum();
    return ans;
  }

  Vector SumMultinomialLogitTransform::from_sum_logits(
      const Vector &sum_and_logits) const {
    double total = sum_and_logits[0];
    ConstVectorView logits(sum_and_logits, 1);

    Vector ans = exp(sum_and_logits);
    ans[0] = 1.0;
    double normalizing_constant = sum(ans);
    ans *= (total / normalizing_constant);
    return ans;
  }

  double SumMultinomialLogitJacobian::element(
      int r, int s, const Vector &positive_numbers) const {
    double total = positive_numbers.sum();
    if (r == 0) {
      return positive_numbers[s] / total;
    } else {
      if (r == s) {
        double prob = positive_numbers[r] / total;
        return total * prob * (1 - prob);
      } else {
        double prob_r = positive_numbers[r] / total;
        double prob_s = positive_numbers[s] / total;
        return -total * prob_r * prob_s;
      }
    }
  }

  // Let 'positive_numbers' denote the original set of positive numbers, and
  // 'sum_and_logits' the values after the transform.  Then the Jacobian matrix
  // is d (positive_numbers) / d (sum_and_logits), arranged with
  // 'positive_numbers' along columns and 'sum_and_logits' along rows.
  //
  // The inverse transform can be written as positive_numbers[s] =
  // sum_and_logits[0] * pi[s], where (for s > 0) pi[s] is the inverse
  // multinomial logit transform of sum_and_logits[s].
  //
  // Note that d pi[s] / d sum_and_logits[r] = pi[s] * (1 - pi[s]) if r == s,
  // and -pi[r] * pi[s] otherwise.
  //
  // Let 'sum' denote the first entry in sum_and_logits, and let eta[s] be entry
  // s in sum_and_logits (for s > 0).  Let nc = 1 + sum(exp(eta)) be the
  // normalizing constant when turning logits back into probabilities.  With
  // this notation, we have the following derivatives.
  //
  // d_positive_numbers[s] / d_sum_and_logits[0] = pi[s]
  // d_positive_numbers[s] / d_sum_and_logits[r >0, r != s] =
  //    - sum * exp(eta[s]) * exp(eta[r]) / (1 + nc)^2
  //   = -sum * pi[s] * pi[r]
  //
  // d_positive_numbers[s] / d_sum_and_logits[s > 0] =
  //     - sum * exp(eta[s]) / nc - exp(eta[s])^2 / nc^2
  //   = -sum (pi[s] - pi[s]^2)
  //
  // In matrix terms this is an asymmetric matrix equal to sum * (diag(pi) - pi
  // * pi') with first row replaced by pi.
  Matrix SumMultinomialLogitJacobian::matrix(
      const Vector &positive_numbers) const {
    int dim = positive_numbers.size();
    Matrix ans(dim, dim, 0.0);
    double total = positive_numbers.sum();
    Vector probs = positive_numbers / total;
    ans.diag() = probs;
    ans.add_outer(probs, probs, -1.0);
    ans *= total;
    ans.row(0) = probs;
    return ans;
  }

  // The log determinant of the invserse Jacobian matrix is easier to derive.
  // The log determinant of the regular Jacobian is simply the negative of the
  // log determinant of the inverse matrix.
  double SumMultinomialLogitJacobian::logdet(
      const Vector &positive_numbers) const {
    return -logdet_inverse_matrix(positive_numbers);
  }

  // d^2 lambda[t] / d eta[r] d eta[s]
  //
  // This is the derivative of the (s,t) element of the Jacobian matrix with
  // respect to eta[r].  Please see comments in front of the 'matrix' member
  // function for hints on the calculus.
  //
  // Because of all the cases that need to be considered, this code is hugely
  // error prone. Checking in the unit tests using JacobianChecker was vital to
  // initial debugging.
  double SumMultinomialLogitJacobian::second_order_element(
      int r, int s, int t, const Vector &positive_numbers) const {
    double total = positive_numbers.sum();
    if (s == 0) {
      // If s == 0 then we're in the first row of the Jacobian matrix, taking
      // the derivative of pi[t] with respect to eta[r].
      if (r == 0) {
        // eta[0] is 'total', which does not appear in the first row of the
        // Jacobian matrix.
        return 0.0;
      } else if (r == t) {
        double prob = positive_numbers[r] / total;
        return prob - prob * prob;
      } else {
        double prob_r = positive_numbers[r] / total;
        double prob_t = positive_numbers[t] / total;
        return -prob_r * prob_t;
      }
    } else {
      // Here s > 0 so the (s, t) entry is eta[0] * (delta(s, t) pi[t] - pi[s] *
      // pi[t]).
      if (s == t) {
        // Derivative of eta[0] * (pi[s] - pi[s]^2) with respect to eta[r].
        if (r == 0) {
          double prob = positive_numbers[s] / total;
          return prob * (1 - prob);
        } else {
          // Case where r > 0
          if (r == s) {
            // derivative of eta[0] * (pi[s] - pi[s]^2) with respect to eta_s.
            double prob = positive_numbers[s] / total;
            double dpi = prob * (1 - prob);
            return total * (dpi - 2 * prob * dpi);
          } else {
            // derivative of eta[0] * (pi[s] - pi[s]^2) with respect to eta_r
            // where r != s.
            double prob_s = positive_numbers[s] / total;
            double prob_r = positive_numbers[r] / total;
            double dpi = -prob_r * prob_s;
            return total * (1 - 2 * prob_s) * dpi;
          }
        }
      } else {
        // Case where s > 0 and s != t
        double prob_s = positive_numbers[s] / total;
        double prob_t = positive_numbers[t] / total;
        if (r == 0) {
          // Derivative of -eta[0] * prob[s] * prob[t] with respect to eta0.
          return -prob_s * prob_t;
        } else if (r == s) {
          // Derivative of -eta[0] * prob[s] * prob[t] with respect to eta[s].
          double dpi_s = prob_s * (1 - prob_s);
          double dpi_t = -prob_s * prob_t;
          return -total * (prob_s * dpi_t + dpi_s * prob_t);
        } else if (r == t) {
          double dpi_s = -prob_s * prob_t;
          double dpi_t = prob_t * (1 - prob_t);
          return -total * (prob_s * dpi_t + dpi_s * prob_t);
        } else {
          double prob_r = positive_numbers[r] / total;
          double dpi_s = -prob_s * prob_r;
          double dpi_t = -prob_t * prob_r;
          return -total * (prob_s * dpi_t + dpi_s * prob_t);
        }
      }
    }
    return negative_infinity();
  }

  // The log determinant of the Jacobian matrix is
  // -log(lam_1 + ... + lam_N) + log(lam_1) + ... + log(lam_N)
  // = -log(eta_0) + log(eta_0 * pi_1) + ... + log(eta_0 * pi_N)
  // Taking derivatives with respect to eta gives..
  //
  // grad[0] = (N-1) / eta_0
  // grad[s > 0] = 1 - N * pi_s
  void SumMultinomialLogitJacobian::add_logdet_gradient(
      Vector &gradient,
      const Vector &positive_numbers) {
    int dim = positive_numbers.size();
    double total = positive_numbers.sum();
    gradient[0] += (dim - 1) / total;
    for (int i = 1; i < dim; ++i) {
      gradient[i] += (1 - dim * positive_numbers[i] / total);
    }
  }

  // See the comments for the gradient.  Taking the derivative of grad[s] with
  // respect to eta[r] is straightforward.  There is no cross derivative of
  // eta[0] with respect to any other component of eta.
  //
  // For the remainder of the elements, use the fact that the derivative of pi_s
  // with respect to eta_r is delta[r,s] * pi_r - pi_r * pi_s.
  void SumMultinomialLogitJacobian::add_logdet_Hessian(
      Matrix &hessian,
      const Vector &positive_numbers) {
    double total = positive_numbers.sum();
    int dim = positive_numbers.size();

    hessian(0, 0) += -(dim - 1) / (total * total);
    for (int r = 1; r < dim; ++r) {
      double prob_r = positive_numbers[r] / total;
      for (int s = 1; s < dim; ++s) {
        double prob_s = positive_numbers[s] / total;
        if (r == s) {
          hessian(r, r) += -dim * (prob_r * (1 - prob_r));
        } else {
          hessian(r, s) += -dim * (-prob_r * prob_s);
        }
      }
    }
  }

  // The inverse of the Jacobian matrix is the Jacobian of the inverse
  // transformation.  We have the following derivatives.
  // * d_eta_0 / d_lambda_r = 1.0 for any r (so the first column of the matrix is
  //     all 1's).
  // * d_eta_s / d_lambda_1 = -1.0 / lambda_1 (for any s > 0).  So the first row
  //     (after the first column) is a constant 1.0 / lambda_1.
  // * deta_s / d_lambda_r (s>0, r>0) = delta(r,s) / lambda_s, so the remaining
  //     diagonal elements are lambda_2, ..., lambda_N.
  //
  // Example from R:
  // > y <- c(1.6484558, 0.1342323, 0.4283595, 1.4628768, 0.9498674)
  //
  // > jake <- function(lambda) {
  //     total <- sum(lambda)
  //     probs <- lambda / total
  //     ans <- total * (diag(probs) - outer(probs, probs))
  //     ans[1, ] <- probs
  //     return(ans)
  //   }
  //
  // > J <- jake(y)
  // > round(solve(J), 8)
  //      [,1]       [,2]       [,3]       [,4]       [,5]
  // [1,]    1 -0.6066283 -0.6066283 -0.6066283 -0.6066283
  // [2,]    1  7.4497732  0.0000000  0.0000000  0.0000000
  // [3,]    1  0.0000000  2.3344878  0.0000000  0.0000000
  // [4,]    1  0.0000000  0.0000000  0.6835846  0.0000000
  // [5,]    1  0.0000000  0.0000000  0.0000000  1.0527785
  Matrix SumMultinomialLogitJacobian::inverse_matrix(
      const Vector &positive_numbers) const {
    int dim = positive_numbers.size();
    Matrix ans(dim, dim, 0.0);
    diag(ans) = 1.0 / positive_numbers;
    ans.row(0) = -1.0 / positive_numbers[0];
    ans.col(0) = 1.0;
    return ans;
  }

  // From expansion by minors on the inverse matrix we get that the determinant is
  //            1/lam2 * 1/lam3 * ... * 1/lamN
  // + 1/lam1 *        * 1/lam3 * ... * 1/lamN
  // + ...
  // + 1/lam1 * 1/lam2 * ... * 1/lam(N-1)
  //
  // This is the product of (1/lamN) times the sum of the reciprocals (i.e. the
  // sum of the lam's).
  //
  // Thus determinant is the sum of the positive numbers times the product of
  // their reciprocals.  Taking logs of this value leads to the code below.
  double SumMultinomialLogitJacobian::logdet_inverse_matrix(
      const Vector &positive_numbers) const {
    double total = sum(positive_numbers);
    double ans = log(total);
    for (int i = 0; i < positive_numbers.size(); ++i) {
      ans -= log(positive_numbers[i]);
    }
    return ans;
  }

}  // namespace BOOM

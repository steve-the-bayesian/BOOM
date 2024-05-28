#ifndef BOOM_TARGETFUN_MULTINOMIAL_LOGIT_TRANSFORM_HPP_
#define BOOM_TARGETFUN_MULTINOMIAL_LOGIT_TRANSFORM_HPP_

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

#include "LinAlg/Vector.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"

#include "TargetFun/Transformation.hpp"

namespace BOOM {

  // A class for handling the Jacobian of the multinomial logit
  // transformation.
  class MultinomialLogitJacobian
      : public Jacobian {
   public:
    // Returns element (r,s) of the Jacobian matrix, which is the
    // derivative of tprobs[s] with respect to logits[r].
    double element(int r, int s, const Vector &truncated_probs) const {
      double ans = -truncated_probs[r] * truncated_probs[s];
      if (r == s) ans += truncated_probs[r];
      return ans;
    }

    // Returns the Jacobian matrix.  If g is the gradient with
    // respect to truncated_probs, then matrix() * g is the gradient
    // with respect to logits.
    Matrix matrix(const Vector &truncated_probs) const override;

    // The log determinant of the Jacobian matrix.  The
    // determinant of the Jacobian matrix is the product of the
    // elements in full_probs, so logdet is the sum of their logs.
    double logdet(const Vector &truncated_probs) const override;

    // Returns the inverse Jacobian matrix, which is both the
    // inverse of matrix() and the Jacobian of the inverse
    // transform.
    //
    // The determinant of the inverse matrix is the product of the
    // elements in 1.0 / full_phi.
    SpdMatrix inverse_matrix(const Vector &truncated_probs) const;

    // Returns the second derivative of tprobs[t] with
    // respect to logits[r] and logits[s].
    // The math:
    //   We start with d_tprobs[t] / d_logits[s]
    //        = delta(s,t) * tprobs[s] - tprobs[s] * tprobs[t]
    //        = J(s, t)
    // where delta(s,t) is the Kronecker delta, and J is the
    // Jacobian matrix.  Then the second derivative is
    //    d2_tprobs[t] / d_logits[r] d_logits[s] =
    //       delta(s,t) * d_tprobs[s] / d_logits[r]
    //          -( d_tprobs[s] / d_logits[r]  * tprobs[t]
    //            + tprobs[s] * d_tprobs[t] / d_logits[r] )
    //       = delta(s,t) * J(r,s)
    //          - (J(r,s) * tprobs[t] + J(r,t) * tprobs[s])
    double second_order_element(
        int r, int s, int t, const Vector &truncated_probs) const override {
      double ans = (s == t) ? element(r, s, truncated_probs) : 0;
      ans -= (element(r, s, truncated_probs) * truncated_probs[t] +
              truncated_probs[s] * element(r, t, truncated_probs));
      return ans;
    }

    // Sets gradient += the gradient of |log(J)| with repect to
    // logits.
    //
    // Args:
    //   gradient: The vector to be incremented by the gradient of |log(J)|.
    //   jacobian_matrix:  The jacobian matrix.
    //   positive: If true then the gradient will be incremented.  Otherwise the
    //     gradient will be decremented.  The false case is useful for the
    //     inverse transformation.
    void add_logits_gradient(const Vector &truncated_probs,
                             Vector &gradient,
                             const SpdMatrix &jacobian_matrix,
                             bool positive = true) const;

    void add_logdet_gradient(Vector &gradient,
                             const Vector &truncated_probs) override;

    // Sets hessian += the hessian of |log(J)| with respect to
    // logits.
    void add_logits_hessian(const Vector &truncated_probs,
                            Matrix &hessian,
                            const SpdMatrix &jacobian_matrix,
                            bool positive = true) const;

    void add_logdet_Hessian(Matrix &hessian,
                            const Vector &truncated_probs) override;

  };  // class Jacobian

  // Maps a discrete probability distribution on the logit scale to the
  // multinomial logit scale.  The first category is the reference class for the
  // transformation, so if 'pi' is a vector of probabilities (summing to 1) then
  // the logit transformation is log(pi / pi[0]).
  //
  // The inverse transformation takes an S-vector of logits ('logits') into and
  // S+1 vector of probabilities ('pi') using exp(logits) / (1 +
  // sum(exp(logits))).
  class MultinomialLogitTransform {
   public:

    // Args:
    //   logits:  The vector of log odds (relative to class 0).
    //   truncated_probs: If true then the returned vector will have the same
    //     dimension as logits.  The elements will sum to less than 1, and the
    //     class probability for class 0 will be left implicit.  If false then
    //     the returned vector has one more element than the input, at the
    //     beginning, filled with the probability of class 0.
    //   truncated_logits: If true then the input logits are free to vary in all
    //     dimensions.  If false, then the leading logit term must be zero.
    Vector to_probs(const Vector &logits, bool truncated_probs = false) const;

    // Transform the un-truncated vector of logits (including the initial
    // element conventionally set to zero) to a same-sized vector of
    // probabilities.
    Vector to_probs_full(const Vector &logits_including_implict_zero) const;

    // Args:
    //   probs:  The vector of probabilities to transform.
    //   truncated: If true then the probs argument is assumed to have its first
    //     element missing, with the remaining elements summing to less than 1.
    //     If false then all elements are assumed present, and the vector should
    //     sum to 1.
    Vector to_logits(const Vector &probs, bool truncated = false) const;

   private:
  };
};

#endif  // BOOM_TARGETFUN_MULTINOMIAL_LOGIT_TRANSFORM_HPP_

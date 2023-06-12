#ifndef BOOM_TARGETFUN_LOGIT_TRANSFORM_HPP_
#define BOOM_TARGETFUN_LOGIT_TRANSFORM_HPP_

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

// The Jacobian base class is in Transformation.hpp
#include "TargetFun/Transformation.hpp"

namespace BOOM {

  // A class for handling the Jacobian of the logit transformation.
  //
  // If z = logit(y) then the Jacobian of the transform is the derivative of y =
  // exp(z) / (1 + exp(z)) with respect to z.  That is exp(z) / (1 + exp(z))^2,
  // which is y * (1 - y).
  class LogitTransformJacobian : public Jacobian {
   public:

    // The log determinant of the Jacobian matrix.
    double logdet(const Vector &probs) const override;

    // Returns the Jacobian matrix.  If g is the gradient with
    // respect to probs, then matrix() * g is the gradient
    // with respect to logits.
    Matrix matrix(const Vector &probs) const override;

    // Returns the second derivative of probs[t] with respect to logits[r] and
    // logits[s].  This is zero unless r == s == t.
    //
    // If y' = y * (1 - y) then y'' can be found by implicit differentiation and
    // the product rule.  y'' = y' * (1 - y) + y * (-y') = y' * (1 - 2y)
    double second_order_element(int r, int s, int t,
                                const Vector &probs) const override {
      if (r == s && s == t) {
        double pq = probs[t] * (1 - probs[t]);
        return pq * (1.0 - 2.0 * probs[t]);
      } else {
        return 0.0;
      }
    }

    // Sets gradient += the gradient of log(|J|) with repect to logits.  Recall
    // that logdet = sum_i log(y[i]).  The gradient of logdet with respect to
    // log(y) is thus a vector of 1's.
    //
    // Args:
    //   probs:
    //   gradient: The vector to be incremented by the gradient of |log(J)|.
    //   positive: If true then the gradient will be incremented.  Otherwise the
    //     gradient will be decremented.  The false case is useful for the
    //     inverse transformation.
    void add_logits_gradient(const Vector &probs,
                             Vector &gradient,
                             bool positive = true) const;

    void add_logdet_gradient(Vector &gradient,
                             const Vector &probs) override {
      add_logits_gradient(probs, gradient, true);
    }

    // Sets hessian += the hessian of log(|J|) with respect to logits.  Recall
    // that the gradient is a vector of 1's, so the Hessian is all 0's.  This
    // function is a no-op.
    void add_logits_hessian(const Vector &probs, Matrix &hessian,
                            bool positive = true) const {}
    void add_logdet_Hessian(Matrix &hessian, const Vector &probs) override {}

  };  // class Jacobian

  class LogitTransform {
   public:
    static Vector transform(const Vector &probs);
    static Vector inverse_transform(const Vector &logits);
  };


};

#endif  // BOOM_TARGETFUN_MULTINOMIAL_LOGIT_TRANSFORM_HPP_

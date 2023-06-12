#ifndef BOOM_TARGETFUN_LOG_TRANSFORM_HPP_
#define BOOM_TARGETFUN_LOG_TRANSFORM_HPP_

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

  // A class for handling the Jacobian of the log transformation.
  //
  // If z = log(y) then the Jacobian of the transform is the derivative of y =
  // exp(z) with respect to z, so exp(z), or just y.
  class LogTransformJacobian : public Jacobian {
   public:

    // The log determinant of the Jacobian matrix.  The determinant is just the
    // product of raw values, so the log determinant is the sum of the logs.
    double logdet(const Vector &raw) const override;

    // Returns the Jacobian matrix.  If g is the gradient with
    // respect to raw, then matrix() * g is the gradient
    // with respect to logs.
    Matrix matrix(const Vector &raw) const override;

    // Returns the inverse Jacobian matrix, which is both the
    // inverse of matrix() and the Jacobian of the inverse
    // transform.
    SpdMatrix inverse_matrix(const Vector &raw) const;

    // Returns the second derivative of raw[t] with respect to logs[r] and
    // logs[s].  This is zero unless r == s == t.  The derivative of y with
    // respect to log(y) is y, no matter how many derivatives you take.
    double second_order_element(int r, int s, int t,
                                const Vector &raw) const override {
      if (r == s && s == t) {
        return raw[t];
      } else {
        return 0.0;
      }
    }

    // Sets gradient += the gradient of log(|J|) with repect to logs.  Recall
    // that logdet = sum_i log(y[i]).  The gradient of logdet with respect to
    // log(y) is thus a vector of 1's.
    //
    // Args:
    //   raw:
    //   gradient: The vector to be incremented by the gradient of |log(J)|.
    //   positive: If true then the gradient will be incremented.  Otherwise the
    //     gradient will be decremented.  The false case is useful for the
    //     inverse transformation.
    void add_logs_gradient(const Vector &raw,
                           Vector &gradient,
                           bool positive = true) const;

    void add_logdet_gradient(Vector &gradient,
                             const Vector &raw) override;

    // Sets hessian += the hessian of log(|J|) with respect to logs.  Recall
    // that the gradient is a vector of 1's, so the Hessian is all 0's.  This
    // function is a no-op.
    void add_logs_hessian(const Vector &raw, Matrix &hessian,
                          bool positive = true) const {}
    void add_logdet_Hessian(Matrix &hessian, const Vector &raw) override {}

  };  // class Jacobian

};

#endif  // BOOM_TARGETFUN_MULTINOMIAL_LOGIT_TRANSFORM_HPP_

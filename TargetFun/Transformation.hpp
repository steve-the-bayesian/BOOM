// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2015 Steven L. Scott

  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License as published by the Free Software Foundation; either
  version 2.1 of the License, or (at your option) any later version.

  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA
*/
#ifndef BOOM_TARGETFUN_TRANSFORMATION_HPP_
#define BOOM_TARGETFUN_TRANSFORMATION_HPP_

#include <functional>
#include <memory>

#include "LinAlg/Matrix.hpp"
#include "LinAlg/SpdMatrix.hpp"
#include "LinAlg/Vector.hpp"

namespace BOOM {

  // The typical use of a Jacobian is to represent the Jacobian of the
  // inverse transformation of a transformed probability density.
  // I.e. if the original distribution has density with respect to x,
  // but we want to conduct inference with respect to z = T(x), then
  // the "matrix" this Jacobian represents is dx / dz.
  //
  //
  class Jacobian {
   public:
    // Default constructor sets preference for original
    // parameterization.
    Jacobian();

    virtual ~Jacobian() {}

    // The log determinant of the Jacobian matrix.  Note the default
    // implementation of this function simply evaluates the
    // determinant of the Jacobian matrix, and then takes the log.
    // This is inefficient and imprecise.  Child classes should
    // overload it if possible.
    virtual double logdet(const Vector &original_params) const;

    // Compute the gradient of the log posterior density with respect
    // to the new parameterization (including the gradient of the log
    // determinant Jacobian, if the second argument is true).
    //
    // Args:
    //   original_gradient: The gradient with respect to the original
    //     parameterization.
    //   add_self_gradient: If true the gradient of log |J| is added
    //     to the return value.
    //
    // Returns:
    //   The gradient with respect to the new parameterization.
    virtual Vector transform_gradient(const Vector &original_gradient,
                                      bool add_self_gradient,
                                      const Vector &original_params);

    // Transforms the Hessian with respect to the original
    // parameterization into the Hessian with respect to the new
    // parameterization.
    //
    // Args:
    //   original_gradient: The gradient with respect to the original
    //     parameterization.
    //   original_Hessian: The Hessian with respect to the original
    //     parameterization.
    //   add_self_Hessian: If true then the Hessian of log |J| will be
    //     added to the return value.
    //
    // Returns:
    //   The Hessian with respect to the new parameterization.
    virtual Matrix transform_Hessian(const Vector &original_gradient,
                                     const Matrix &original_Hessian,
                                     bool add_self_Hessian,
                                     const Vector &original_params);

    // Returns the Jacobian matrix.  If g is the gradient with respect
    // to the original parameterization, then matrix() * g is the
    // gradient with respect to the new parameterization.
    //
    // The preferred implementation here is for concrete classes to
    // keep a flag that that keeps track of whether the Jacobian
    // matrix is current.  The flag is turned off by a call to
    // evaluate() with the matrix created as needed.
    //
    // The Jacobian matrix is organized so that "old parameterization"
    // corresponds to columns and "new parameterization" to rows.  That
    // way matrix() * gradient transforms the gradient with respect to
    // the old parameterization into the gradient with respect to the
    // new one.
    virtual Matrix matrix(const Vector &original_params) const = 0;

    // Returns the second derivative of original_parameterization[t] with
    // respect to new_parameterization[r] and new_parameterization[s].
    // Said another way, second_order_element(r,s,t) is the derivative
    // with respect to new_parameterization[r] of the (s,t) element of
    // the Jacobian matrix.
    //
    // This function is used to implement transform_Hessian (through
    // transform_second_order_gradient).  If either of those functions
    // is overloaded in such as way as to not need this one, then this
    // one can be a no-op.
    virtual double second_order_element(
        int r, int s, int t,
        const Vector &original_params) const = 0;

    // Take the working_hessian, and add the second order gradient
    // term to element (r,s).  That is,
    // working_hessian(r, s) += sum_i
    //     original_gradient[i] * d^2 original_parameterization[i] /
    //              d new_parameterization[r], d new_parameterization[s].
    //
    // This is the second term in the second order chain rule.
    //
    // NOTE: The default implementation of this function knows nothing
    // of sparsity.  It is cubic in the dimension of the
    // transformation, and very expensive.
    virtual void transform_second_order_gradient(
        SpdMatrix &working_hessian,
        const Vector &original_gradient,
        const Vector &original_params);

    // Sets gradient += the gradient of |log(J)| with respect to
    // new_parameterization.
    virtual void add_logdet_gradient(Vector &gradient,
                                     const Vector &original_params) = 0;

    // Sets hessian += the hessian of |log(J)| with respect to
    // new_parameterization.
    virtual void add_logdet_Hessian(Matrix &hessian,
                                    const Vector &original_params) = 0;

  };  // class Jacobian

  // A Transformation is a twice differentiable mapping from an
  // original parameterization x to a new parameterization z.  A
  // Transformation converts a log density in x, and replaces it with
  // the equivalent log density in z.
  class Transformation {
   public:
    typedef std::function<double(const Vector &, Vector &, Matrix &, uint)>
        Target;
    typedef std::function<Vector(const Vector &)> Mapping;

    // Args:
    //   log_density_old_parameterization: A log density function in the
    //     original parameterization.
    //   inverse_mapping: A mapping back from the transformed space to the
    //     original parameterization.
    //   jacobian:  A Jacobian object
    Transformation(const Target &log_density_old_parameterization,
                   const Mapping &inverse_mapping,
                   Jacobian *jacobian);

    double operator()(const Vector &new_parameterization, Vector &gradient,
                      Matrix &hessian, uint nderiv) const;
    double operator()(const Vector &new_parameterization) const;
    double operator()(const Vector &new_parameterization,
                      Vector &gradient) const;
    double operator()(const Vector &new_parameterization, Vector &gradient,
                      Matrix &hessian) const;

   private:
    Target logp_original_scale_;

    // x = inverse_mapping_(z)
    Mapping inverse_mapping_;

    // dx / dz
    std::shared_ptr<Jacobian> jacobian_;
  };

}  // namespace BOOM

#endif  //  BOOM_TARGETFUN_TRANSFORMATION_HPP_

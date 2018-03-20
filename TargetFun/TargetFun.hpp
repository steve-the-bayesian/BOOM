// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005 Steven L. Scott

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

#ifndef TARGET_FUN_H
#define TARGET_FUN_H

#include <functional>
#include "LinAlg/Vector.hpp"
#include "cpputil/Ptr.hpp"
#include "cpputil/RefCounted.hpp"

namespace BOOM {
  // A suite of function object which can be passed to optimization
  // routines.
  class TargetFun : private RefCounted {
   public:
    virtual double operator()(const Vector &x) const = 0;
    ~TargetFun() override {}
    friend void intrusive_ptr_add_ref(TargetFun *);
    friend void intrusive_ptr_release(TargetFun *);
  };

  void intrusive_ptr_add_ref(TargetFun *);
  void intrusive_ptr_release(TargetFun *);

  //----------------------------------------------------------------------
  class dTargetFun : virtual public TargetFun {
   public:
    double operator()(const Vector &x) const override = 0;
    virtual double operator()(const Vector &x, Vector &g) const = 0;
  };

  //----------------------------------------------------------------------
  //  A dTargetFun that can take a scalar partial derivative with
  //  respect to a just one coordinate.
  class dScalarEnabledTargetFun : public dTargetFun {
   public:
    // Return the function value at x, while computing the partial
    // derivative with respect to x[position].
    virtual double scalar_derivative(const Vector &x, double &derivative,
                                     int position) const = 0;
  };

  //----------------------------------------------------------------------
  // A dTargetFun that can also take second derivatives.
  class d2TargetFun : virtual public dTargetFun {
   public:
    virtual double operator()(const Vector &x, Vector &g, Matrix &h,
                              uint nderiv) const = 0;
    double operator()(const Vector &x) const override;
    double operator()(const Vector &x, Vector &g) const override;
    virtual double operator()(const Vector &x, Vector &g, Matrix &h) const;
  };

  //----------------------------------------------------------------------
  // A common (and superior) pattern for writing functions and
  // derivatives is to pass the derivatives as pointers.  This class
  // is an adapter to convert that pattern to the one expected by
  // various function optimizers.
  //
  // This object evaluates to the sum of one or more functions with
  // the signature specified by TargetType.  The function arguments
  // are as follows.
  //   x: The function argument.
  //   gradient: If non-NULL the gradient is computed and output
  //     here.  If NULL then no derivative computations are made.
  //   Hessian: If Hessian and gradient are both non-NULL the
  //     Hessian is computed and output here.  If NULL then the
  //     Hessian is not computed.
  //   reset_derivatives: If true then a non-NULL gradient or
  //     Hessian will be resized and set to zero.  If false then a
  //     non-NULL gradient or Hessian will have derivatives of
  //     log-liklihood added to its input value.  It is an error if
  //     reset_derivatives is false and the wrong-sized non-NULL
  //     argument is passed.
  class d2TargetFunPointerAdapter : public d2TargetFun {
   public:
    typedef std::function<double(const Vector &x, Vector *gradient,
                                 Matrix *Hessian, bool reset_derivatives)>
        TargetType;
    d2TargetFunPointerAdapter() {}
    explicit d2TargetFunPointerAdapter(const TargetType &target);
    d2TargetFunPointerAdapter(const TargetType &prior,
                              const TargetType &likelihood);
    void add_function(const TargetType &target);
    double operator()(const Vector &x, Vector &gradient, Matrix &Hessian,
                      uint nderiv) const override;
    using d2TargetFun::operator();

    // If targets_ is empty then an error is reported (e.g. by
    // throwing an exception).
    void check_not_empty() const;

   private:
    std::vector<TargetType> targets_;
  };

  //======================================================================
  //
  class ScalarTargetFun : private RefCounted {
   public:
    virtual double operator()(double x) const = 0;
    ~ScalarTargetFun() override {}
    friend void intrusive_ptr_add_ref(ScalarTargetFun *);
    friend void intrusive_ptr_release(ScalarTargetFun *);
  };
  void intrusive_ptr_add_ref(ScalarTargetFun *);
  void intrusive_ptr_release(ScalarTargetFun *);
  //----------------------------------------------------------------------
  class dScalarTargetFun : virtual public ScalarTargetFun {
   public:
    double operator()(double x) const override = 0;
    virtual double operator()(double x, double &d) const = 0;
  };
  //----------------------------------------------------------------------
  class d2ScalarTargetFun : virtual public dScalarTargetFun {
   public:
    virtual double operator()(double x, double &d1, double &d2,
                              uint nderiv) const = 0;
    double operator()(double x) const override {
      double d1, d2;
      return (*this)(x, d1, d2, 0);
    }
    double operator()(double x, double &d) const override {
      double d2;
      return (*this)(x, d, d2, 1);
    }
    virtual double operator()(double x, double &g, double &h) const {
      return (*this)(x, g, h, 2);
    }
  };

  //======================================================================
  // Turn a TargetFun into a ScalarTargetFun.  (I.e. turns a function
  // that takes a vector argument into one that takes a scalar
  // argument).
  class ScalarTargetFunAdapter : public ScalarTargetFun {
   public:
    ScalarTargetFunAdapter(const std::function<double(const Vector &)> &F,
                           Vector *X, uint position);
    double operator()(double x) const override;

   private:
    std::function<double(const Vector &)> f_;
    Vector *wsp_;
    uint which_;
  };

  // Turns a dScalarEnabledTargetFun into a dScalarTargetFun.
  // (I.e. turns a function that takes a vector argument into one that
  // takes a scalar argument).
  class dScalarTargetFunAdapter : public dScalarTargetFun {
   public:
    dScalarTargetFunAdapter(const Ptr<dScalarEnabledTargetFun> &f, Vector *x,
                            uint position);
    double operator()(double x) const override;
    double operator()(double x, double &derivative) const override;

   private:
    Ptr<dScalarEnabledTargetFun> f_;
    Vector *x_;
    uint position_;
  };

}  // namespace BOOM
#endif  // TARGET_FUN_H

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

#include "TargetFun/TargetFun.hpp"
#include <cmath>
#include <functional>
#include "LinAlg/Matrix.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  void intrusive_ptr_add_ref(TargetFun *s) { s->up_count(); }
  void intrusive_ptr_release(TargetFun *s) {
    s->down_count();
    if (s->ref_count() == 0) delete s;
  }

  double d2TargetFun::operator()(const Vector &x) const {
    Vector g;
    Matrix h;
    return (*this)(x, g, h, 0);
  }
  double d2TargetFun::operator()(const Vector &x, Vector &g) const {
    Matrix h;
    return (*this)(x, g, h, 1);
  }
  double d2TargetFun::operator()(const Vector &x, Vector &g, Matrix &h) const {
    return (*this)(x, g, h, 2);
  }

  //======================================================================
  void intrusive_ptr_add_ref(ScalarTargetFun *s) { s->up_count(); }
  void intrusive_ptr_release(ScalarTargetFun *s) {
    s->down_count();
    if (s->ref_count() == 0) delete s;
  }
  //----------------------------------------------------------------------

  d2TargetFunPointerAdapter::d2TargetFunPointerAdapter(
      const TargetType &target) {
    add_function(target);
  }

  d2TargetFunPointerAdapter::d2TargetFunPointerAdapter(
      const TargetType &prior, const TargetType &likelihood) {
    add_function(prior);
    add_function(likelihood);
  }

  void d2TargetFunPointerAdapter::add_function(const TargetType &fun) {
    targets_.push_back(fun);
  }

  double d2TargetFunPointerAdapter::operator()(const Vector &x, Vector &g,
                                               Matrix &h, uint nderiv) const {
    check_not_empty();
    double ans = targets_[0](x, nderiv > 0 ? &g : nullptr,
                             nderiv > 1 ? &h : nullptr, true);
    for (int i = 1; i < targets_.size(); ++i) {
      ans += targets_[i](x, nderiv > 0 ? &g : nullptr,
                         nderiv > 1 ? &h : nullptr, false);
    }
    return ans;
  }

  void d2TargetFunPointerAdapter::check_not_empty() const {
    if (targets_.empty()) {
      report_error(
          "Error in d2TargetFunPointerAdapter.  "
          "No component functions specified.");
    }
  }

  //======================================================================

  ScalarTargetFunAdapter::ScalarTargetFunAdapter(
      const std::function<double(const Vector &)> &F, Vector *X, uint position)
      : f_(F), wsp_(X), which_(position) {}

  double ScalarTargetFunAdapter::operator()(double x) const {
    (*wsp_)[which_] = x;
    return f_(*wsp_);
  }

  //======================================================================
  dScalarTargetFunAdapter::dScalarTargetFunAdapter(
      const Ptr<dScalarEnabledTargetFun> &f, Vector *x, uint position)
      : f_(f), x_(x), position_(position) {}

  double dScalarTargetFunAdapter::operator()(double x_arg) const {
    (*x_)[position_] = x_arg;
    return (*f_)(*x_);
  }

  double dScalarTargetFunAdapter::operator()(double x_arg,
                                             double &derivative) const {
    (*x_)[position_] = x_arg;
    return f_->scalar_derivative(*x_, derivative, position_);
  }
}  // namespace BOOM

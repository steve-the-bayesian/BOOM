// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2006 Steven L. Scott

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
#include "Models/ConstrainedVectorParams.hpp"
#include <algorithm>

namespace BOOM {

  using VC = BOOM::VectorConstraint;
  using EC = BOOM::ElementConstraint;
  EC::ElementConstraint(uint el, double val) : element_(el), value_(val) {}

  bool EC::check(const Vector &v) const { return v[element_] == value_; }
  void EC::impose(Vector &v) const { v[element_] = value_; }
  Vector EC::expand(const Vector &v) const {
    Vector ans(v.size() + 1);
    Vector::const_iterator b(v.begin()), e(v.end());
    Vector::const_iterator pos = b + element_;
    std::copy(b, pos, ans.begin());
    std::copy(pos, e, ans.begin() + element_ + 1);
    impose(ans);
    return ans;
  }

  Vector EC::reduce(const Vector &v) const {
    if (v.empty()) {
      return Vector(0);
    }
    Vector ans(v.size() - 1);
    Vector::const_iterator b(v.begin()), e(v.end());
    std::copy(b, b + element_, ans.begin());
    std::copy(b + element_ + 1, e, ans.begin() + element_);
    return ans;
  }
  //------------------------------------------------------------
  using SC = BOOM::SumConstraint;
  SC::SumConstraint(double x) : sum_(x) {}

  bool SC::check(const Vector &v) const { return v.sum() == sum_; }

  void SC::impose(Vector &v) const {
    double tot = v.sum();
    v.back() = sum_ - tot;
  }

  Vector SC::expand(const Vector &v) const {
    Vector ans(v.size() + 1);
    std::copy(v.begin(), v.end(), ans.begin());
    impose(ans);
    return ans;
  }

  Vector SC::reduce(const Vector &v) const {
    Vector ans(v.begin(), v.end() - 1);
    return ans;
  }

  //============================================================

  using CVP = BOOM::ConstrainedVectorParams;

  CVP::ConstrainedVectorParams(uint p, double x, const Ptr<VC> &vc)
      : VectorParams(p, x), c_(vc) {
    if (!vc) {
      c_ = new NoConstraint;
    }
  }

  CVP::ConstrainedVectorParams(const Vector &v, const Ptr<VC> &vc)
      : VectorParams(v), c_(vc) {
    if (!vc) {
      c_ = new NoConstraint;
    }
  }

  CVP::ConstrainedVectorParams(const CVP &rhs)
      : Data(rhs), Params(rhs), VectorParams(rhs), c_(rhs.c_) {}

  CVP *CVP::clone() const { return new CVP(*this); }

  Vector CVP::vectorize(bool minimal) const {
    if (minimal) {
      return c_->reduce(value());
    }
    return value();
  }

  Vector::const_iterator CVP::unvectorize(Vector::const_iterator &v,
                                          bool minimal) {
    Vector tmp(vectorize(minimal));
    Vector::const_iterator e = v + tmp.size();
    tmp.assign(v, e);
    if (minimal) {
      set(c_->expand(tmp));
    } else {
      set(tmp);
    }
    return e;
  }

  Vector::const_iterator CVP::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator b(v.begin());
    return unvectorize(b, minimal);
  }

  //============================================================

}  // namespace BOOM

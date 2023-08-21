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
#include "cpputil/report_error.hpp"
#include <algorithm>

namespace BOOM {

  ElementConstraint::ElementConstraint(uint el, double val)
      : element_(el),
        value_(val)
  {}

  bool ElementConstraint::check(const Vector &v) const {
    return v[element_] == value_;
  }

  Vector &ElementConstraint::impose(Vector &v) const {
    v[element_] = value_;
    return v;
  }

  Vector ElementConstraint::expand(const Vector &v) const {
    Vector ans(v.size() + 1);
    Vector::const_iterator b(v.begin()), e(v.end());
    Vector::const_iterator pos = b + element_;
    std::copy(b, pos, ans.begin());
    std::copy(pos, e, ans.begin() + element_ + 1);
    impose(ans);
    return ans;
  }

  Vector ElementConstraint::reduce(const Vector &v) const {
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

  Vector &SC::impose(Vector &v) const {
    double tot = v.sum();
    v.back() = sum_ - tot;
    return v;
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

  //===========================================================================

  bool ProportionalSumConstraint::check(const Vector &v) const {
    return (fabs(v.sum() - sum_) < 1e-5);
  }

  Vector &ProportionalSumConstraint::impose(Vector &v) const {
    double total = v.sum();
    if (fabs(total - sum_) > 1e-5) {
      v *= sum_ / total;
    }
    return v;
  }

  Vector ProportionalSumConstraint::expand(const Vector &constrained) const {
    double value = sum_ - constrained.sum();
    Vector ans = concat(value, constrained);
    return impose(ans);
  }

  Vector ProportionalSumConstraint::reduce(const Vector &full) const {
    return Vector(ConstVectorView(full, 1));
  }

  //===========================================================================
  ConstrainedVectorParams::ConstrainedVectorParams(
      const Vector &v, const Ptr<VectorConstraint> &constraint)
      : VectorParams(v), constraint_(constraint)
  {
    if (!constraint) {
      constraint_ = new NoConstraint;
    }
    Vector value(v);
    constraint_->impose(value);
    bool signal_observers = false;
    VectorParams::set(value, signal_observers);
  }

  ConstrainedVectorParams::ConstrainedVectorParams(
      const ConstrainedVectorParams &rhs)
      : Data(rhs),
        Params(rhs),
        VectorParams(rhs),
        constraint_(rhs.constraint_)
  {}

  ConstrainedVectorParams *ConstrainedVectorParams::clone() const {
    return new ConstrainedVectorParams(*this);
  }

  uint ConstrainedVectorParams::size(bool minimal) const {
    uint ans = VectorParams::size();
    if (minimal) {
      ans -= constraint_->minimal_size_reduction();
    }
    return ans;
  }

  Vector ConstrainedVectorParams::vectorize(bool minimal) const {
    if (minimal) {
      return constraint_->reduce(value());
    }
    return value();
  }

  Vector::const_iterator ConstrainedVectorParams::unvectorize(
      Vector::const_iterator &v, bool minimal) {
    Vector tmp(vectorize(minimal));
    Vector::const_iterator e = v + tmp.size();
    tmp.assign(v, e);
    if (minimal) {
      set(constraint_->expand(tmp));
    } else {
      set(tmp);
    }
    return e;
  }

  void ConstrainedVectorParams::set(const Vector &value, bool signal_change) {
    int n = value.size();
    if (n == size(true)) {
      VectorParams::set(constraint_->expand(value), signal_change);
    } else if (n == size(false)) {
      Vector val = value;
      VectorParams::set(constraint_->impose(val), signal_change);
    } else {
      report_error("Wrong size argument.");
    }
  }

}  // namespace BOOM

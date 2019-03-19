// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2007 Steven L. Scott

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

#include "Models/UniformModel.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"
#include "cpputil/math_utils.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    typedef UniformSuf US;
    typedef UniformModel UM;
  }  // namespace

  US::UniformSuf() : lo_(BOOM::infinity()), hi_(BOOM::negative_infinity()) {}

  US::UniformSuf(double a, double b) : lo_(a), hi_(b) {
    assert(a <= b && "Arguments out of order in UniformSuf constructor");
  }

  US::UniformSuf(const std::vector<double> &rhs) {
    lo_ = rhs[0];
    hi_ = rhs[0];
    uint n = rhs.size();
    for (uint i = 1; i < n; ++i) {
      double x = rhs[i];
      if (x < lo_) lo_ = x;
      if (x > hi_) hi_ = x;
    }
  }

  US::UniformSuf(const US &rhs)
      : Sufstat(rhs), SufTraits(rhs), lo_(rhs.lo_), hi_(rhs.hi_) {}

  US *US::clone() const { return new US(*this); }

  void US::clear() {
    lo_ = BOOM::infinity();
    hi_ = BOOM::negative_infinity();
  }

  void US::update_raw(double x) {
    lo_ = x < lo_ ? x : lo_;
    hi_ = x > hi_ ? x : hi_;
  }

  void US::Update(const DoubleData &d) { update_raw(d.value()); }

  double US::lo() const { return lo_; }
  double US::hi() const { return hi_; }
  void US::set_lo(double a) {
    lo_ = a;
    assert(hi_ >= lo_);
  }
  void US::set_hi(double b) {
    hi_ = b;
    assert(hi_ >= lo_);
  }

  void US::combine(const Ptr<US> &s) {
    lo_ = std::min<double>(lo_, s->lo_);
    hi_ = std::max<double>(hi_, s->hi_);
  }

  void US::combine(const US &s) {
    lo_ = std::min<double>(lo_, s.lo_);
    hi_ = std::max<double>(hi_, s.hi_);
  }

  UniformSuf *US::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector US::vectorize(bool) const {
    Vector ans(2);
    ans[0] = lo_;
    ans[1] = hi_;
    return ans;
  }

  Vector::const_iterator US::unvectorize(Vector::const_iterator &v, bool) {
    lo_ = *v;
    ++v;
    hi_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator US::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator it = v.begin();
    return unvectorize(it, minimal);
  }

  std::ostream &US::print(std::ostream &out) const {
    return out << lo_ << " " << hi_;
  }

  //======================================================================
  UM::UniformModel(double a, double b)
      : ParamPolicy(new UnivParams(a), new UnivParams(b)), DataPolicy(new US) {}

  UM::UniformModel(const std::vector<double> &data)
      : ParamPolicy(new UnivParams(0), new UnivParams(1)),
        DataPolicy(new US(data)) {
    mle();
  }

  UM::UniformModel(const UM &rhs)
      : Model(rhs),
        ParamPolicy(rhs),
        DataPolicy(rhs),
        PriorPolicy(rhs),
        DiffDoubleModel(rhs),
        LoglikeModel(rhs) {}

  UM *UM::clone() const { return new UM(*this); }

  double UM::lo() const { return LoParam()->value(); }
  double UM::hi() const { return HiParam()->value(); }
  double UM::nc() const { return 1.0 / (hi() - lo()); }

  void UM::set_lo(double a) {
    LoParam()->set(a);
    assert(a <= hi());
  }
  void UM::set_hi(double b) {
    HiParam()->set(b);
    assert(b >= lo());
  }

  void UM::set_ab(double a, double b) {
    assert(a <= b);
    LoParam()->set(a);
    HiParam()->set(b);
  }

  double UM::mean() const { return .5 * (lo() + hi()); }

  double UM::variance() const { return square(hi() - lo()) / 12.0; }

  Ptr<UnivParams> UM::LoParam() { return ParamPolicy::prm1(); }
  Ptr<UnivParams> UM::HiParam() { return ParamPolicy::prm2(); }
  const Ptr<UnivParams> UM::LoParam() const { return ParamPolicy::prm1(); }
  const Ptr<UnivParams> UM::HiParam() const { return ParamPolicy::prm2(); }

  double UM::Logp(double x, double &g, double &h, uint nd) const {
    bool outside = x > hi() || x < lo();
    if (nd > 0) {
      g = 0;
      if (nd > 1) h = 0;
    }
    return outside ? BOOM::negative_infinity() : log(nc());
  }

  double UM::loglike(const Vector &ab) const {
    double lo = ab[0];
    double hi = ab[1];
    bool hi_ok = suf()->hi() <= hi;
    bool lo_ok = suf()->lo() >= lo;
    if (hi_ok && lo_ok) return log(nc());
    return BOOM::negative_infinity();
  }

  void UM::mle() { set_ab(suf()->lo(), suf()->hi()); }

  double UM::sim(RNG &rng) const { return runif_mt(rng, lo(), hi()); }

}  // namespace BOOM

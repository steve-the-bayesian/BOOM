// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2012 Steven L. Scott

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

#include "Models/WeightedGaussianSuf.hpp"
#include "Models/SufstatAbstractCombineImpl.hpp"

namespace BOOM {

  namespace {
    typedef WeightedGaussianSuf WGS;
  }

  WGS::WeightedGaussianSuf(double sum, double sumsq, double n, double sumw)
      : sum_(sum), sumsq_(sumsq), n_(n), sumw_(sumw) {}

  WeightedGaussianSuf *WGS::clone() const {
    return new WeightedGaussianSuf(*this);
  }

  void WGS::clear() { sum_ = sumsq_ = n_ = sumw_ = 0; }

  void WGS::Update(const WeightedDoubleData &data) {
    update_raw(data.value(), data.weight());
  }

  void WGS::add_mixture_data(double y, double w, double prob) {
    sumw_ += w * prob;
    sum_ += y * w * prob;
    sumsq_ += y * y * w * prob;
    n_ += prob;
  }

  void WGS::update_raw(double y, double w) {
    sumw_ += w;
    sum_ += y * w;
    sumsq_ += y * y * w;
    n_ += 1.0;
  }

  void WGS::combine(const Ptr<WeightedGaussianSuf> &suf) { combine(*suf); }

  void WGS::combine(const WeightedGaussianSuf &rhs) {
    sum_ += rhs.sum();
    sumsq_ += rhs.sumsq();
    n_ += rhs.n();
    sumw_ += rhs.sumw();
  }

  WeightedGaussianSuf *WGS::abstract_combine(Sufstat *s) {
    return abstract_combine_impl(this, s);
  }

  Vector WGS::vectorize(bool) const {
    Vector ans(4);
    ans[0] = n_;
    ans[1] = sum_;
    ans[2] = sumsq_;
    ans[3] = sumw_;
    return ans;
  }

  Vector::const_iterator WGS::unvectorize(Vector::const_iterator &v, bool) {
    n_ = *v;
    ++v;
    sum_ = *v;
    ++v;
    sumsq_ = *v;
    ++v;
    sumw_ = *v;
    ++v;
    return v;
  }

  Vector::const_iterator WGS::unvectorize(const Vector &v, bool minimal) {
    Vector::const_iterator b = v.begin();
    return unvectorize(b, minimal);
  }

  std::ostream &WGS::print(std::ostream &out) const {
    out << "n      = " << n_ << endl
        << "sum_   = " << sum_ << endl
        << "sumsq_ = " << sumsq_ << endl
        << "sumw_  = " << sumw_ << endl;
    return out;
  }

}  // namespace BOOM

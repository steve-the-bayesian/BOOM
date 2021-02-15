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
#include <algorithm>
#include <string>

#include "LinAlg/VectorView.hpp"
#include "Models/ParamTypes.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  Vector vectorize(const std::vector<Ptr<Params>> &v, bool minimal) {
    uint N = v.size();
    uint vec_size(0);

    for (uint i = 0; i < N; ++i) vec_size += v[i]->size(minimal);
    Vector ans(vec_size);
    Vector::iterator it = ans.begin();
    for (uint i = 0; i < N; ++i) {
      Vector tmp = v[i]->vectorize(minimal);
      it = std::copy(tmp.begin(), tmp.end(), it);
    }
    return ans;
  }
  void unvectorize(std::vector<Ptr<Params>> &pvec,
                   const Vector &v,
                   bool minimal) {
    Vector::const_iterator it = v.begin();
    for (uint i = 0; i < pvec.size(); ++i) {
      it = pvec[i]->unvectorize(it, minimal);
    }
  }

  std::ostream &operator<<(std::ostream &out,
                           const std::vector<Ptr<Params>> &v) {
    out << vectorize(v, false);
    return out;
  }

  Params::Params() {}

  Params::Params(const Params &rhs) : Data(rhs) {}

  //======================================================================

  typedef UnivData<double> UDD;
  UnivParams::UnivParams() : Params(), UDD(0) {}
  UnivParams::UnivParams(double x) : UDD(x) {}
  UnivParams::UnivParams(const UnivParams &rhs)
      : Data(rhs), Params(rhs), UDD(rhs) {}
  UnivParams *UnivParams::clone() const { return new UnivParams(*this); }

  Vector UnivParams::vectorize(bool) const {
    Vector ans(1);
    ans[0] = value();
    return ans;
  }
  Vector::const_iterator UnivParams::unvectorize(Vector::const_iterator &v,
                                                 bool) {
    set(*v);
    return ++v;
  }
  Vector::const_iterator UnivParams::unvectorize(const Vector &v, bool) {
    Vector::const_iterator b = v.begin();
    return unvectorize(b);
  }

  void UnivParamsObserver::set(const double &rhs, bool Signal) {
    report_error("set is disabled.");
  }

  //============================================================
  typedef VectorData VD;

  VectorParams::VectorParams(uint p, double x) : VD(p, x) {}

  VectorParams::VectorParams(const Vector &v) : VD(v) {}

  VectorParams::VectorParams(const VectorParams &rhs)
      : Data(rhs), Params(rhs), VD(rhs) {}

  VectorParams *VectorParams::clone() const { return new VectorParams(*this); }

  uint VectorParams::size(bool) const { return dim(); }

  Vector VectorParams::vectorize(bool) const { return value(); }

  Vector::const_iterator VectorParams::unvectorize(Vector::const_iterator &v,
                                                   bool) {
    Vector::const_iterator e = v + size(false);
    Vector tmp(v, e);
    set(tmp);
    return e;
  }

  Vector::const_iterator VectorParams::unvectorize(const Vector &v, bool) {
    Vector::const_iterator b = v.begin();
    return unvectorize(b);
  }

  //============================================================
  typedef MatrixData MD;
  typedef MatrixParams MP;

  MP::MatrixParams(uint r, uint c, double x) : MD(r, c, x) {}

  MP::MatrixParams(const Matrix &m) : MD(m) {}

  MP::MatrixParams(const MatrixParams &rhs) : Data(rhs), Params(rhs), MD(rhs) {}

  MatrixParams *MP::clone() const { return new MP(*this); }

  uint MP::size(bool) const { return value().size(); }

  Vector MP::vectorize(bool) const {
    Vector ans(value().begin(), value().end());
    return ans;
  }

  Vector::const_iterator MP::unvectorize(Vector::const_iterator &b, bool) {
    Vector::const_iterator e = b + size();
    const Matrix &val(value());
    Matrix tmp(b, e, val.nrow(), val.ncol());
    set(tmp);
    return e;
  }
  Vector::const_iterator MP::unvectorize(const Vector &v, bool) {
    Vector::const_iterator b = v.begin();
    return unvectorize(b);
  }

}  // namespace BOOM

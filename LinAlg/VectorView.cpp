// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2018 Steven L. Scott

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
// std library includes
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include "LinAlg/EigenMap.hpp"
#include "LinAlg/Matrix.hpp"
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorView.hpp"

#include "cpputil/math_utils.hpp"
#include "cpputil/report_error.hpp"
#include "distributions.hpp"

namespace BOOM {

  namespace {
    template <class V1, class V2>
    double dot_impl(const V1 &v1, const V2 &v2) {
      if (v1.stride() > 0 && v2.stride() > 0) {
        return EigenMap(v1).dot(EigenMap(v2));
      } else {
        // Strides can be negative for vector views that have been reversed.
        double ans = 0;
        for (int i  = 0; i < v1.size(); ++i) {
          ans += v1[i] * v2[i];
        }
        return ans;
      }
    }
  }  // namespace

  typedef VectorView VV;

  VV::iterator VV::begin() { return iterator(V, V, stride()); }
  VV::iterator VV::end() {
    return iterator(V + size() * stride(), V, stride());
  }
  VV::const_iterator VV::begin() const {
    return const_iterator(V, V, stride());
  }
  VV::const_iterator VV::end() const {
    return const_iterator(V + size() * stride(), V, stride());
  }

  VV::reverse_iterator VV::rbegin() {
    return std::reverse_iterator<iterator>(begin());
  }
  VV::reverse_iterator VV::rend() {
    return std::reverse_iterator<iterator>(end());
  }
  VV::const_reverse_iterator VV::rbegin() const {
    return std::reverse_iterator<const_iterator>(begin());
  }
  VV::const_reverse_iterator VV::rend() const {
    return std::reverse_iterator<const_iterator>(end());
  }

  VV::VectorView(double *first, uint n, int s)
      : V(first), nelem_(n), stride_(s) {}

  VV &VV::reset(double *first, uint n, uint s) {
    V = first;
    nelem_ = n;
    stride_ = s;
    return *this;
  }

  VV::VectorView(Vector &v, uint first)
      : V(v.data() + first), nelem_(v.size() - first), stride_( 1) {
    // Allow first to be zero in the case of empty vectors, so we can can have a
    // VectorView into an empty.
    if ( (first > 0) && (first >= v.size())) {
      report_error("First element in view is past the end of the hosting "
                   "vector.");
    }
  }

  VV::VectorView(Vector &v, uint first, uint length)
      : V(v.data() + first), nelem_(length), stride_(1) {
    if (v.size() < first + length) {
      report_error("Vector is not large enough to host the requested view.");
    }
  }

  VV::VectorView(VectorView v, uint first)
      : V(v.data() + first * v.stride()),
        nelem_(v.size() - first),
        stride_(v.stride()) {}

  VV::VectorView(VectorView v, uint first, uint length)
      : V(v.data() + first * v.stride()), nelem_(length), stride_(v.stride()) {}

  VV &VV::operator=(double x) {
    std::fill(begin(), end(), x);
    return *this;
  }

  VV &VV::operator=(const Vector &x) {
    assert(x.size() == size());
    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  VV &VV::operator=(const VectorView &x) {
    assert(x.size() == size());
    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  VV &VV::operator=(const ConstVectorView &x) {
    assert(x.size() == size());
    std::copy(x.begin(), x.end(), begin());
    return *this;
  }

  void VV::randomize() {
    uint n = size();
    double *d = data();
    for (uint i = 0; i < n; ++i) d[i] = runif(0, 1);
  }

  VV &VV::operator+=(const double &x) {
    VV &A(*this);
    for (uint i = 0; i < size(); ++i) A[i] += x;
    return *this;
  }

  VV &VV::operator-=(const double &x) {
    VV &A(*this);
    for (uint i = 0; i < size(); ++i) A[i] -= x;
    return *this;
  }

  VV &VV::operator*=(const double &x) {
    EigenMap(*this) *= x;
    //    dscal(size(), x, data(), stride());
    return *this;
  }

  VV &VV::operator/=(const double &x) {
    assert(x != 0.0);
    EigenMap(*this) /= x;
    //    dscal(size(), 1.0/x, data(), stride());
    return *this;
  }

  VV &VV::operator+=(const VectorView &y) {
    assert(y.size() == size());
    EigenMap(*this) += EigenMap(y);
    // daxpy(size(), 1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::operator+=(const ConstVectorView &y) {
    assert(y.size() == size());
    EigenMap(*this) += EigenMap(y);
    // daxpy(size(), 1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::operator+=(const Vector &y) {
    assert(y.size() == size());
    EigenMap(*this) += EigenMap(y);
    //    daxpy(size(), 1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::operator-=(const Vector &y) {
    assert(y.size() == size());
    EigenMap(*this) -= EigenMap(y);
    //    daxpy(size(), -1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::operator-=(const VectorView &y) {
    assert(y.size() == size());
    EigenMap(*this) -= EigenMap(y);
    //    daxpy(size(), -1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::operator-=(const ConstVectorView &y) {
    assert(y.size() == size());
    EigenMap(*this) -= EigenMap(y);
    // daxpy(size(), -1.0, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::axpy(const Vector &y, double a) {
    assert(y.size() == size());
    EigenMap(*this) += a * EigenMap(y);
    //    daxpy(size(), a, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::axpy(const VectorView &y, double a) {
    assert(y.size() == size());
    EigenMap(*this) += a * EigenMap(y);
    //    daxpy(size(), a, y.data(), y.stride(), data(), stride());
    return *this;
  }

  VV &VV::axpy(const ConstVectorView &y, double a) {
    assert(y.size() == size());
    EigenMap(*this).array() += a * EigenMap(y).array();
    //    daxpy(size(), a, y.data(), y.stride(), data(), stride());
    return *this;
  }

  namespace {
    inline void dmul(uint n, double *x, uint xs, const double *y, uint ys) {
      for (uint i = 0; i < n; ++i) {
        *x *= *y;
        x += xs;
        y += ys;
      }
    }
    inline void ddiv(uint n, double *x, uint xs, const double *y, uint ys) {
      for (uint i = 0; i < n; ++i) {
        *x /= *y;
        x += xs;
        y += ys;
      }
    }
    inline double mul(double x, double y) { return x * y; }
  }  // namespace

  VV &VV::operator*=(const Vector &y) {
    assert(size() == y.size());
    dmul(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV &VV::operator*=(const VectorView &y) {
    assert(size() == y.size());
    dmul(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV &VV::operator*=(const ConstVectorView &y) {
    assert(size() == y.size());
    dmul(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV &VV::operator/=(const Vector &y) {
    assert(size() == y.size());
    ddiv(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV &VV::operator/=(const VectorView &y) {
    assert(size() == y.size());
    ddiv(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  VV &VV::operator/=(const ConstVectorView &y) {
    assert(size() == y.size());
    ddiv(size(), data(), stride(), y.data(), y.stride());
    return *this;
  }

  double VV::normsq() const { return EigenMap(*this).squaredNorm(); }

  double VV::normalize_prob() {
    double s = sum();
    if (s == 0) {
      report_error("normalizing constant is zero in VV::normalize_logprob");
    }
    operator/=(s);
    return s;
  }

  double VV::normalize_logprob() {
    double nc = 0;
    VectorView &x = *this;
    double m = max();
    uint n = size();
    for (uint i = 0; i < n; ++i) {
      x[i] = std::exp(x[i] - m);
      nc += x[i];
    }
    x /= nc;
    return nc;  // might want to change this
  }

  double VV::min() const {
    const_iterator it = std::min_element(begin(), end());
    return *it;
  }

  double VV::max() const {
    const_iterator it = std::max_element(begin(), end());
    return *it;
  }

  uint VV::imax() const {
    const_iterator it = std::max_element(begin(), end());
    return it - begin();
  }

  uint VV::imin() const {
    const_iterator it = std::min_element(begin(), end());
    return it - begin();
  }

  double VV::sum() const { return std::accumulate(begin(), end(), 0.0); }

  double VV::abs_norm() const { return EigenMap(*this).lpNorm<1>(); }

  double VV::prod() const { return std::accumulate(begin(), end(), 1.0, mul); }

  double VV::dot(const Vector &y) const { return dot_impl(*this, y); }
  double VV::dot(const VectorView &y) const { return dot_impl(*this, y); }
  double VV::dot(const ConstVectorView &y) const { return dot_impl(*this, y); }

  namespace {
    template <class V1, class V2>
    double affdot_impl(const V1 &x, const V2 &y) {
      uint n = x.size();
      uint m = y.size();
      if (m == n) {
        return x.dot(y);
      } else if (m == n + 1) {
        return y[0] + ConstVectorView(y, 1).dot(x);
      } else if (n == m + 1) {
        return x[0] + ConstVectorView(x, 1).dot(y);
      } else {
        report_error("x and y do not conform in affdot.");
        return negative_infinity();
      }
    }
  }  // namespace

  double VV::affdot(const Vector &y) const { return affdot_impl(*this, y); }

  double VV::affdot(const VectorView &y) const { return affdot_impl(*this, y); }

  VV &VV::transform(const std::function<double(double)> &f) {
    for (int i = 0; i < size(); ++i) {
      double *d = V + i * stride_;
      *d = f(*d);
    }
    return *this;
  }

  std::ostream &operator<<(std::ostream &out, const VV &v) {
    for (uint i = 0; i < v.size(); ++i) out << v[i] << " ";
    return out;
  }

  void print(const VectorView &v) { std::cout << v << std::endl; }

  void print(const ConstVectorView &v) { std::cout << v << std::endl; }

  istream &operator<<(istream &in, VV &v) {
    for (uint i = 0; i < v.size(); ++i) in >> v[i];
    return in;
  }

  //======================================================================

  typedef ConstVectorView CVV;

  CVV::const_iterator CVV::begin() const {
    return const_iterator(V, V, stride());
  }

  CVV::const_iterator CVV::end() const {
    return const_iterator(V + size() * stride(), V, stride());
  }

  CVV::const_reverse_iterator CVV::rbegin() const {
    return std::reverse_iterator<const_iterator>(begin());
  }

  CVV::const_reverse_iterator CVV::rend() const {
    return std::reverse_iterator<const_iterator>(end());
  }

  CVV::ConstVectorView(const double *first_element, uint n, int s)
      : V(first_element), nelem_(n), stride_(s) {}

  CVV::ConstVectorView(const Vector &v, uint first_element)
      : V(v.data() + first_element),
        nelem_(v.size() - first_element),
        stride_(1) {}

  CVV::ConstVectorView(const Vector &v, uint first_element, uint length)
      : V(v.data() + first_element), nelem_(length), stride_(1) {}

  CVV::ConstVectorView(const CVV &v, uint first_element)
      : V(v.data() + first_element * v.stride()),
        nelem_(v.size() - first_element),
        stride_(v.stride()) {}

  CVV::ConstVectorView(const std::vector<double> &v, uint first_element)
      : V(v.data() + first_element),
        nelem_(v.size() - first_element),
        stride_(1) {}

  CVV::ConstVectorView(const VectorView &v, uint first_element, uint length)
      : V(v.data() + first_element * v.stride()),
        nelem_(length),
        stride_(v.stride()) {}

  CVV::ConstVectorView(const CVV &v, uint first_element, uint length)
      : V(v.data() + first_element * v.stride()),
        nelem_(length),
        stride_(v.stride()) {}

  CVV::ConstVectorView(const VectorView &v, uint first_element)
      : V(v.data() + first_element * v.stride()),
        nelem_(v.size() - first_element),
        stride_(v.stride()) {}

  CVV::ConstVectorView(const std::vector<double> &v, uint first_element,
                       uint length)
      : V(v.data() + first_element), nelem_(length), stride_(1) {}

  double CVV::normsq() const { return EigenMap(*this).squaredNorm(); }

  double CVV::min() const {
    const_iterator it = std::min_element(begin(), end());
    return *it;
  }

  double CVV::max() const {
    const_iterator it = std::max_element(begin(), end());
    return *it;
  }

  uint CVV::imax() const {
    const_iterator it = std::max_element(begin(), end());
    return it - begin();
  }

  uint CVV::imin() const {
    const_iterator it = std::min_element(begin(), end());
    return it - begin();
  }

  double CVV::sum() const { return std::accumulate(begin(), end(), 0.0); }

  double CVV::abs_norm() const { return EigenMap(*this).lpNorm<1>(); }

  double CVV::prod() const { return std::accumulate(begin(), end(), 1.0, mul); }

  double CVV::dot(const Vector &y) const { return dot_impl(*this, y); }
  double CVV::dot(const VectorView &y) const { return dot_impl(*this, y); }
  double CVV::dot(const ConstVectorView &y) const { return dot_impl(*this, y); }
  double CVV::affdot(const Vector &y) const { return affdot_impl(*this, y); }
  double CVV::affdot(const VectorView &y) const {
    return affdot_impl(*this, y);
  }
  double CVV::affdot(const ConstVectorView &y) const {
    return affdot_impl(*this, y);
  }

  CVV CVV::reverse() const {
    const double *start = V + (nelem_ - 1) * stride_;
    return CVV(start, nelem_, -stride_);
  }

  std::ostream &operator<<(std::ostream &out, const CVV &v) {
    for (uint i = 0; i < v.size(); ++i) out << v[i] << " ";
    return out;
  }

  namespace {
    template <class VECTOR>
    VectorView tail_impl(VECTOR &v, int size) {
      if (v.size() <= size) {
        return VectorView(v);
      }
      int n = v.size();
      return VectorView(v, n - size);
    }

    template <class VECTOR>
    ConstVectorView const_tail_impl(const VECTOR &v, int size) {
      const ConstVectorView view(v);
      if (v.size() <= size) {
        return ConstVectorView(view);
      }
      int n = view.size();
      return ConstVectorView(view, n - size);
    }
  }  // namespace

  VectorView tail(Vector &v, int size) { return tail_impl(v, size); }
  VectorView tail(VectorView &v, int size) { return tail_impl(v, size); }

  ConstVectorView const_tail(const Vector &v, int size) {
    return const_tail_impl(v, size);
  }
  ConstVectorView const_tail(const VectorView &v, int size) {
    return const_tail_impl(v, size);
  }
  ConstVectorView const_tail(const ConstVectorView &v, int size) {
    return const_tail_impl(v, size);
  }

}  // namespace BOOM

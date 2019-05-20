// Copyright 2018 Google LLC. All Rights Reserved.
/*
  Copyright (C) 2005-2010 Steven L. Scott

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

#include "Models/StateSpace/Filters/SparseVector.hpp"
#include <iostream>
#include "LinAlg/SpdMatrix.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  typedef SparseVectorReturnProxy SVRP;
  typedef std::map<int, double>::iterator It;
  typedef std::map<int, double>::const_iterator Cit;

  SVRP::SparseVectorReturnProxy(int position, double value, SparseVector *v)
      : position_(position), value_(value), v_(v) {}

  SVRP &SVRP::operator=(double x) {
    v_->elements_[position_] = x;
    value_ = x;
    return *this;
  }

  SVRP::operator double() const { return value_; }

  //======================================================================
  SparseVector::SparseVector(int n) : size_(n) {
    if (n < 0) {
      report_error("SparseVector initialized with a negative size.");
    }
    size_ = n;
  }

  SparseVector::SparseVector(const Vector &dense) : size_(dense.size()) {
    for (int i = 0; i < size_; ++i) {
      elements_[i] = dense[i];
    }
  }

  int SparseVector::size() const { return size_; }

  SparseVector &SparseVector::concatenate(const SparseVector &rhs) {
    for (Cit it = rhs.elements_.begin(); it != rhs.elements_.end(); ++it) {
      int indx = size_ + it->first;
      elements_[indx] = it->second;
    }
    size_ += rhs.size_;
    return *this;
  }

  double SparseVector::operator[](int n) const {
    check_index(n);
    Cit it = elements_.find(n);
    if (it == elements_.end()) return 0;
    return it->second;
  }

  SparseVectorReturnProxy SparseVector::operator[](int n) {
    check_index(n);
    It it = elements_.find(n);
    if (it == elements_.end()) {
      return SparseVectorReturnProxy(n, 0, this);
    }
    return SparseVectorReturnProxy(n, it->second, this);
  }

  void SparseVector::check_index(int n) const {
    if (n < 0) {
      report_error("SparseVector indexed with a negative value");
    } else if (n > size_) {
      report_error("Access past the end of SparseVector");
    }
  }

  SparseVector &SparseVector::operator*=(double x) {
    for (It it = elements_.begin(); it != elements_.end(); ++it)
      it->second *= x;
    return *this;
  }

  SparseVector &SparseVector::operator/=(double x) {
    return (*this) *= 1.0 / x;
  }

  double SparseVector::sum() const {
    double ans = 0;
    for (Cit it = elements_.begin(); it != elements_.end(); ++it)
      ans += it->second;
    return ans;
  }

  template <class VEC>
  double do_dot(const VEC &v, const std::map<int, double> &m, int size) {
    if (v.size() != size) {
      std::ostringstream err;
      err << "incompatible vector in SparseVector dot product: \n"
          << "dense vector: " << v << "\n";
      for (const auto &el : m) {
        err << "sparse[" << el.first << "] = " << el.second << "\n";
      }
      report_error(err.str());
    }
    double ans = 0;
    for (Cit it = m.begin(); it != m.end(); ++it)
      ans += it->second * v[it->first];
    return ans;
  }
  double SparseVector::dot(const Vector &v) const {
    return do_dot(v, elements_, size_);
  }
  double SparseVector::dot(const VectorView &v) const {
    return do_dot(v, elements_, size_);
  }
  double SparseVector::dot(const ConstVectorView &v) const {
    return do_dot(v, elements_, size_);
  }

  double SparseVector::sandwich(const SpdMatrix &P) const {
    double ans = 0;
    for (const auto &row : elements_) {
      int i = row.first;
      double xi = row.second;
      for (const auto &col : elements_) {
        int j = col.first;
        double xj = col.second;
        double increment = xi * xj * P(i, j);
        if (j == i) {
          ans += increment;
          break;
        }
        ans += 2 * increment;
      }
    }
    return ans;
  }

  Matrix SparseVector::outer_product_transpose(const Vector &x,
                                               double scale) const {
    Matrix ans(x.size(), this->size(), 0.0);
    for (const auto &el : elements_) {
      int i = el.first;
      ans.col(i) = x;
      ans.col(i) *= (el.second * scale);
    }
    return ans;
  }

  Vector SparseVector::dense() const {
    Vector ans(size(), 0.0);
    for (Cit it = elements_.begin(); it != elements_.end(); ++it) {
      ans[it->first] = it->second;
    }
    return ans;
  }

  void SparseVector::add_this_to(Vector &x, double weight) const {
    if (x.size() != size_) {
      ostringstream err;
      err << "SparseVector::add_this_to called with incompatible x:" << endl
          << "this->size() = " << size_ << endl
          << "x.size()     = " << x.size() << endl;
      report_error(err.str());
    }
    for (Cit it = elements_.begin(); it != elements_.end(); ++it) {
      x[it->first] += weight * it->second;
    }
  }

  void SparseVector::add_this_to(VectorView x, double weight) const {
    if (x.size() != size_) {
      ostringstream err;
      err << "SparseVector::add_this_to called with incompatible x:" << endl
          << "this->size() = " << size_ << endl
          << "x.size()     = " << x.size() << endl;
      report_error(err.str());
    }
    for (Cit it = elements_.begin(); it != elements_.end(); ++it) {
      x[it->first] += weight * it->second;
    }
  }

  void SparseVector::add_outer_product(SpdMatrix &m, double scale) const {
    for (const auto &i : elements_) {
      for (const auto &j : elements_) {
        m(i.first, j.first) += i.second * j.second * scale;
      }
    }
  }

  //======================================================================

  Vector operator*(const SpdMatrix &P, const SparseVector &z) {
    int n = nrow(P);
    Vector ans(n);
    for (int i = 0; i < n; ++i) {
      //      ans[i] = z.dot(P.col(i));
      ans[i] = z.dot(P.row(i));
    }
    return ans;
  }

  Vector operator*(SubMatrix P, const SparseVector &z) {
    int n = P.nrow();
    Vector ans(n);
    for (int i = 0; i < n; ++i) {
      ans[i] = z.dot(P.row(i));
    }
    return ans;
  }

  std::ostream &operator<<(std::ostream &out, const SparseVector &z) {
    int n = z.size();
    if (n == 0) return out;
    out << z[0];
    for (int i = 1; i < n; ++i) out << " " << z[i];
    return out;
  }

}  // namespace BOOM

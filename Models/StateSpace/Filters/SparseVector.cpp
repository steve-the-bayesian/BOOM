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
#include "LinAlg/SpdMatrix.hpp"
#include "cpputil/report_error.hpp"

namespace BOOM {

  typedef SparseVectorReturnProxy SVRP;
  typedef SparseVectorViewReturnProxy SVVRP;
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

  SVVRP::SparseVectorViewReturnProxy(
      int position_in_view, double value, SparseVectorView *view)
      : position_in_base_vector_(
            view->position_in_base_vector(position_in_view)),
        value_(value),
        v_(view) {}

  SVVRP &SVVRP::operator=(double x) {
    v_->base_vector_->elements_[position_in_base_vector_] = x;
    value_ = x;

    if (position_in_base_vector_ < v_->begin_->first) {
      v_->begin_ = v_->base_vector_->elements_.find(position_in_base_vector_);
      v_->cbegin_ = v_->begin_;
    }

    if (position_in_base_vector_ > v_->end_->first) {
      v_->end_ = v_->base_vector_->elements_.find(position_in_base_vector_);
      v_->cend_ = v_->end_;
    }
    return *this;
  }

  SVVRP &SVVRP::operator+=(double increment) {
    v_->base_vector_->elements_[position_in_base_vector_] += increment;
    value_ += increment;
    return *this;
  }

  SVVRP &SVVRP::operator-=(double decrement) {
    v_->base_vector_->elements_[position_in_base_vector_] -= decrement;
    value_ -= decrement;
    return *this;
  }

  SVVRP &SVVRP::operator*=(double scale) {
    v_->base_vector_->elements_[position_in_base_vector_] *= scale;
    value_ *= scale;
    return *this;
  }

  SVVRP &SVVRP::operator/=(double scale) {
    v_->base_vector_->elements_[position_in_base_vector_] /= scale;
    value_ /= scale;
    return *this;
  }

  SVVRP::operator double() const {return value_;}

  //======================================================================
  SparseVector::SparseVector(size_t n) : size_(n) {
    size_ = n;
  }

  SparseVector::SparseVector(const Vector &dense) : size_(dense.size()) {
    for (int i = 0; i < size_; ++i) {
      elements_[i] = dense[i];
    }
  }

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

  template <class VEC, class ITERATOR>
  double do_dot(const VEC &v, ITERATOR begin, ITERATOR end, int size) {
    if (v.size() != size) {
      std::ostringstream err;
      err << "incompatible vector in SparseVector dot product: \n"
          << "dense vector: " << v << "\n";
      for (ITERATOR it = begin; it != end; ++it) {
        err << "sparse[" << it->first << "] = " << it->second << "\n";
      }
      report_error(err.str());
    }
    double ans = 0;
    for (ITERATOR it = begin; it != end; ++it)
      ans += it->second * v[it->first];
    return ans;
  }

  template <class VEC>
  double do_dot(const VEC &v, const std::map<int, double> &m, int size) {
    return do_dot(v, m.begin(), m.end(), size);
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

  //===========================================================================
  SparseVectorView::SparseVectorView(
      SparseVector *base_vector,
      size_t start,
      size_t size,
      int stride)
      : base_vector_(base_vector),
        start_(start),
        size_(size),
        stride_(stride),
        begin_(this),
        end_(this),
        cbegin_(this),
        cend_(this)
  {
    initialize_iterators();
  }

  double SparseVectorView::operator[](int n) const {
    return (*base_vector_)[position_in_base_vector(n)];
  }

  SparseVectorViewReturnProxy SparseVectorView::operator[](int n) {
    return SparseVectorViewReturnProxy(
        n, (*base_vector_)[position_in_base_vector(n)], this);
  }

  SparseVectorView &SparseVectorView::operator*=(double x) {
    for (auto it = begin(); it != end(); ++it) {
      it->second *= x;
    }
    return *this;
  }

  SparseVectorView &SparseVectorView::operator/=(double x) {
    for (auto it = begin(); it != end(); ++it) {
      it->second /= x;
    }
    return *this;
  }

  double SparseVectorView::sum() const {
    double ans = 0;
    for (auto it = begin(); it != end(); ++it) {
      ans += it->second;
    }
    return ans;
  }

  double SparseVectorView::dot(const Vector &x) const {
    return do_dot(x, begin(), end(), size());
  }

  double SparseVectorView::dot(const VectorView &x) const {
    return do_dot(x, begin(), end(), size());
  }

  double SparseVectorView::dot(const ConstVectorView &x) const {
    return do_dot(x, begin(), end(), size());
  }

  // bool SparseVectorView::operator==(const SparseVector &rhs) const {
  //   return (size() == rhs.size() && std::equal(
  //       begin(),
  //       end(),
  //       rhs.begin(),
  //       [this](std::pair<int, double> &lhs,
  //              std::pair<int, double> &rhs) {
  //         return lhs.second == rhs.second && lhs.first == rhs.first + this->start_;
  //       }));
  // }

  bool SparseVectorView::operator==(const SparseVectorView &rhs) const {
    if (size() != rhs.size()) {
      return false;
    } else {
      auto rhs_it = rhs.begin();
      for (auto it = begin(); it != end(); ++it, ++rhs_it) {
        if ((it->first != rhs_it->first) || (it->second != rhs_it->second)) {
          return false;
        }
      }
    }
    return true;
  }

  void SparseVectorView::add_this_to(Vector &x, double weight) const {
    if (x.size() != size_) {
      ostringstream err;
      err << "SparseVectorView::add_this_to called with incompatible x:" << endl
          << "this->size() = " << size_ << endl
          << "x.size()     = " << x.size() << endl;
      report_error(err.str());
    }
    for (auto it = begin(); it != end(); ++it) {
      x[position_in_view(it->first)] += weight * it->second;
    }
  }

  void SparseVectorView::add_this_to(VectorView x, double weight) const {
    if (x.size() != size_) {
      ostringstream err;
      err << "SparseVectorView::add_this_to called with incompatible x:" << endl
          << "this->size() = " << size_ << endl
          << "x.size()     = " << x.size() << endl;
      report_error(err.str());
    }
    for (auto it = begin(); it != end(); ++it) {
      x[position_in_view(it->first)] += weight * it->second;
    }
  }

  void SparseVectorView::add_outer_product(SpdMatrix &m, double scale) const {
    for (auto row_it = begin(); row_it != end(); ++row_it) {
      int I = position_in_view(row_it->first);
      for (auto col_it = begin(); col_it != end(); ++col_it) {
        int J = position_in_view(col_it->first);
        m(I, J) += row_it->second * col_it->second * scale;
      }
    }
  }

  double SparseVectorView::sandwich(const SpdMatrix &P) const {
    double ans = 0;
    for (auto row_it = begin(); row_it != end(); ++row_it) {
      int i = position_in_view(row_it->first);
      double xi = row_it->second;
      for (auto col_it = begin(); col_it != end(); ++col_it) {
        int j = position_in_view(col_it->first);
        int xj = col_it->second;
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

  Matrix SparseVectorView::outer_product_transpose(
      const Vector &x, double scale) const {
    Matrix ans(x.size(), this->size(), 0.0);
    for (auto it = begin(); it != end(); ++it) {
      int j = position_in_view(it->first);
      ans.col(j) = x * (it->second * scale);
    }
    return ans;
  }

  Vector SparseVectorView::dense() const {
    Vector ans(size(), 0.0);
    for (auto it = begin(); it != end(); ++it) {
      ans[position_in_view(it->first)] = it->second;
    }
    return ans;
  }

  SparseVectorViewIterator SparseVectorView::begin() {
    return begin_;
  }

  SparseVectorViewConstIterator SparseVectorView::begin() const{
    return cbegin_;
  }

  SparseVectorViewIterator SparseVectorView::end() {
    return end_;
  }

  SparseVectorViewConstIterator SparseVectorView::end() const {
    return cend_;
  }

  void SparseVectorView::initialize_iterators() {
    size_t start = position_in_base_vector(0);
    size_t final = position_in_base_vector(size() - 1);
    begin_.set_underlying_iterator(base_vector_->elements_.lower_bound(start));
    end_.set_underlying_iterator(base_vector_->elements_.upper_bound(final));
    cbegin_ = begin_;
    cend_ = end_;
  }

}  // namespace BOOM

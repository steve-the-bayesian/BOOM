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

#ifndef BOOM_NEWLA_VECTOR_VIEW_HPP
#define BOOM_NEWLA_VECTOR_VIEW_HPP

#include <cassert>
#include <iosfwd>
#include <iterator>
#include <string>
#include "LinAlg/Vector.hpp"
#include "LinAlg/VectorViewIterator.hpp"

namespace BOOM {
  class Vector;
  class SpdMatrix;
  class Matrix;

  class VectorView {
   public:
    typedef VectorViewIterator iterator;
    typedef VectorViewConstIterator const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    //--------- constructors, destructor, assigment, operator== ----------
    VectorView(double *first_elem, uint Nelem, int Stride);

    // View from a specified first element until the end.
    explicit VectorView(Vector &v, uint first = 0);
    // No default argument for 'first', to distinguish from copy constructor.
    VectorView(VectorView v, uint first);

    // Contiguous view for 'len' elements, starting from 'first'.
    VectorView(Vector &v, uint first, uint len);
    VectorView(VectorView v, uint first, uint len);

    VectorView &operator=(double x);
    VectorView &operator=(const Vector &x);
    VectorView &operator=(const VectorView &x);
    VectorView &operator=(const ConstVectorView &x);

    VectorView &reset(double *first_elem, uint Nelem, uint Stride);

    void randomize();  // fills the Vector with U(0,1) random numbers

    //-------------- STL Vector stuff ---------------------
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

    reverse_iterator rbegin();
    reverse_iterator rend();
    const_reverse_iterator rbegin() const;
    const_reverse_iterator rend() const;

    double *data() { return V; }
    const double *data() const { return V; }
    int stride() const { return stride_; }

    uint size() const { return nelem_; }    // returns number of elements
    uint length() const { return nelem_; }  // same as size()
    bool empty() const { return nelem_ == 0; }

    //------------------ subscripting -----------------------
    const double &operator[](int n) const {
      assert(n >= 0 && n < nelem_);
      return *(V + n * stride_);
    }
    double &operator[](int n) {
      assert(n >= 0 && n < nelem_);
      return *(V + n * stride_);
    }
    const double &operator()(int n) const {
      assert(n >= 0 && n < nelem_);
      return *(V + n * stride_);
    }
    double &operator()(int n) {
      assert(n >= 0 && n < nelem_);
      return *(V + n * stride_);
    }
    double &front() { return *V; }
    const double &front() const { return *V; }
    double &back() { return *(V + (nelem_ - 1) * stride_); }
    const double &back() const { return *(V + (nelem_ - 1) * stride_); }

    //---------------- input/output -------------------------
    std::ostream &write(std::ostream &, bool endl = true) const;
    std::istream &read(std::istream &);

    //--------- math ----------------
    VectorView &operator+=(const double &x);
    VectorView &operator-=(const double &x);
    VectorView &operator*=(const double &x);
    VectorView &operator/=(const double &x);

    VectorView &operator+=(const Vector &y);
    VectorView &operator-=(const Vector &y);
    VectorView &operator*=(const Vector &y);
    VectorView &operator/=(const Vector &y);

    VectorView &operator+=(const VectorView &y);
    VectorView &operator-=(const VectorView &y);
    VectorView &operator*=(const VectorView &y);
    VectorView &operator/=(const VectorView &y);

    VectorView &operator+=(const ConstVectorView &y);
    VectorView &operator-=(const ConstVectorView &y);
    VectorView &operator*=(const ConstVectorView &y);
    VectorView &operator/=(const ConstVectorView &y);

    VectorView &axpy(const Vector &y, double a = 1.0);
    VectorView &axpy(const VectorView &y, double a = 1.0);
    VectorView &axpy(const ConstVectorView &y, double a = 1.0);

    double normsq() const;
    double normalize_prob();
    double normalize_logprob();
    double min() const;
    double max() const;
    uint imax() const;  // index of maximal/minmal element
    uint imin() const;
    double sum() const;
    double prod() const;
    double abs_norm() const;

    double dot(const Vector &y) const;
    double dot(const VectorView &y) const;
    double dot(const ConstVectorView &y) const;

    double affdot(const Vector &y) const;
    double affdot(const VectorView &y) const;
    // affine dot product:  dim(y) == dim(x)-1. ignores lower bounds

    VectorView &transform(const std::function<double(double)> &f);

   private:
    double *V;
    uint nelem_;
    int stride_;
    bool inrange(uint n) const { return n < nelem_; }
  };

  // IO
  std::ostream &operator<<(std::ostream &out, const VectorView &x);
  // prints to stdout.  This function is here so it can be called from gdb.
  void print(const VectorView &v);
  std::istream &operator>>(std::istream &, VectorView &);
  // Vector view size is known from construction

  template <class VEC>
  VectorView subvector(VEC &v, uint start) {
    assert(start < v.size());
    return VectorView(v.data() + start, v.size() - start, v.stride());
  }

  template <class VEC>
  VectorView subvector(VEC &v, uint start, uint stop) {
    assert(start <= stop && start < v.size());
    uint size = 1 + stop - start;
    return VectorView(v.data() + start, size, v.stride());
  }

  //======================================================================
  class ConstVectorView {
    const double *V;
    uint nelem_;
    int stride_;

    bool inrange(uint n) const { return n < nelem_; }

   public:
    typedef VectorViewConstIterator const_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    //--------- constructors, destructor, assigment, operator== ----------

    // View an aribtrary chunk of memory
    ConstVectorView(const double *first_elem, uint Nelem, int Stride);

    // View from first_element to the end.  These constructors are
    // non-explicit because I want automatic conversions from Vector
    // and VectorView to ConstVectorView to be legal.
    // cppcheck-suppress noExplicitConstructor
    ConstVectorView(const Vector &v, uint first_element = 0);
    // cppcheck-suppress noExplicitConstructor
    ConstVectorView(const VectorView &rhs, uint first_element = 0);
    // cppcheck-suppress noExplicitConstructor
    ConstVectorView(const ConstVectorView &v, uint first_element = 0);
    // cppcheck-suppress noExplicitConstructor
    ConstVectorView(const std::vector<double> &v, uint first_element = 0);

    // View from first_element to first_element + length - 1
    ConstVectorView(const Vector &v, uint first_element, uint length);
    ConstVectorView(const VectorView &v, uint first_element, uint length);
    ConstVectorView(const ConstVectorView &v, uint first_element, uint length);
    ConstVectorView(const std::vector<double> &v, uint first_element,
                    uint length);

    template <class V>
    explicit ConstVectorView(const V &rhs)
        : V(rhs.data()), nelem_(rhs.size()), stride_(rhs.stride()) {}

    //    ConstVectorView(const ConstVectorView &rhs) = default;
    ConstVectorView(ConstVectorView &&rhs) = default;

    //-------------- STL Vector stuff ---------------------
    const_iterator begin() const;
    const_iterator end() const;
    const_reverse_iterator rbegin() const;
    const_reverse_iterator rend() const;

    const double *data() const { return V; }
    int stride() const { return stride_; }

    uint size() const { return nelem_; }    // returns number of elements
    uint length() const { return nelem_; }  // same as size()
    bool empty() const { return nelem_ == 0; }

    //------------------ subscripting -----------------------
    const double &operator[](int n) const {
      assert(n >= 0 && n < nelem_);
      return *(V + n * stride_);
    }
    const double &operator()(int n) const {
      assert(n >= 0 && n < nelem_);
      return *(V + n * stride_);
    }
    const double &front() const { return *V; }
    const double &back() const { return *(V + (nelem_ - 1) * stride_); }

    //---------------- input/output -------------------------
    std::ostream &write(std::ostream &, bool endl = true) const;

    //--------- math ----------------
    double normsq() const;
    double min() const;
    double max() const;
    uint imax() const;  // index of maximal/minmal element
    uint imin() const;
    double sum() const;
    double abs_norm() const;
    double prod() const;

    double dot(const Vector &y) const;
    double dot(const VectorView &y) const;
    double dot(const ConstVectorView &y) const;

    double affdot(const Vector &y) const;
    double affdot(const VectorView &y) const;
    double affdot(const ConstVectorView &y) const;
    // affine dot product:  dim(y) == dim(x)-1. ignores lower bounds

    // Returns a ConstVectorView that points to the same elements as
    // *this, but in reverse order.  This is done by pointing to the
    // last element in the CVV, keeping the same length, and using a
    // negative stride.
    ConstVectorView reverse() const;
  };

  inline double sum(const ConstVectorView &view) { return view.sum(); }
  inline double sum(const VectorView &view) { return view.sum(); }
  template <class VEC>
  double sumlog(const VEC &v) {
    double ans = 0;
    for (int i = 0; i < v.size(); ++i) {
      ans += log(v[i]);
    }
    return ans;
  }

  // IO
  std::ostream &operator<<(std::ostream &out, const ConstVectorView &x);
  // prints to stdout.  This function is here so it can be called from gdb.
  void print(const ConstVectorView &v);

  template <class VEC>
  ConstVectorView subvector(const VEC &v, uint start) {
    return ConstVectorView(v.data() + start, v.size() - start, v.stride());
  }

  template <class VEC>
  ConstVectorView subvector(const VEC &v, uint start, uint stop) {
    assert(start <= stop && start < v.size());
    uint size = 1 + stop - start;
    return ConstVectorView(v.data() + start, size, v.stride());
  }

  // Field operations for VectorView.  Interactions with double and Vector are
  // defined in Vector.cpp
  // Addition
  inline Vector operator+(const ConstVectorView &x, const ConstVectorView &y) {
    Vector ans(x);
    ans += y;
    return ans;
  }
  inline Vector operator+(const ConstVectorView &x, const VectorView &y) {
    return x + ConstVectorView(y);
  }
  inline Vector operator+(const VectorView &x, const ConstVectorView &y) {
    return ConstVectorView(x) + y;
  }
  inline Vector operator+(const VectorView &x, const VectorView &y) {
    return ConstVectorView(x) + ConstVectorView(y);
  }

  // Subtraction
  inline Vector operator-(const ConstVectorView &x, const ConstVectorView &y) {
    Vector ans(x);
    ans -= y;
    return ans;
  }
  inline Vector operator-(const VectorView &x, const VectorView &y) {
    return ConstVectorView(x) - ConstVectorView(y);
  }
  inline Vector operator-(const ConstVectorView &x, const VectorView &y) {
    return x - ConstVectorView(y);
  }
  inline Vector operator-(const VectorView &x, const ConstVectorView &y) {
    return ConstVectorView(x) - y;
  }

  // Multipliplication
  Vector operator*(double x, const VectorView &v);
  Vector operator*(double x, const ConstVectorView &v);
  Vector operator*(const VectorView &v, double x);
  Vector operator*(const ConstVectorView &v, double x);
  inline Vector operator*(const ConstVectorView &x, const ConstVectorView &y) {
    Vector ans(x);
    ans *= y;
    return ans;
  }
  inline Vector operator*(const VectorView &x, const VectorView &y) {
    return ConstVectorView(x) * ConstVectorView(y);
  }
  inline Vector operator*(const ConstVectorView &x, const VectorView &y) {
    return x * ConstVectorView(y);
  }
  inline Vector operator*(const VectorView &x, const ConstVectorView &y) {
    return ConstVectorView(x) * y;
  }

  // Division
  Vector operator/(double x, const VectorView &v);
  Vector operator/(double x, const ConstVectorView &v);
  Vector operator/(const VectorView &v, double x);
  Vector operator/(const ConstVectorView &v, double x);
  inline Vector operator/(const ConstVectorView &x, const ConstVectorView &y) {
    Vector ans(x);
    ans /= y;
    return ans;
  }
  inline Vector operator/(const VectorView &x, const VectorView &y) {
    return ConstVectorView(x) / ConstVectorView(y);
  }
  inline Vector operator/(const ConstVectorView &x, const VectorView &y) {
    return x / ConstVectorView(y);
  }
  inline Vector operator/(const VectorView &x, const ConstVectorView &y) {
    return ConstVectorView(x) / y;
  }

  // Return a view giving the last 'size' elements of the vector or view.  If
  // the size of the vector or view is less than 'size' then return a view to
  // the first argument.
  VectorView tail(Vector &v, int size);
  VectorView tail(VectorView &v, int size);
  ConstVectorView const_tail(const Vector &v, int size);
  ConstVectorView const_tail(const VectorView &v, int size);
  ConstVectorView const_tail(const ConstVectorView &v, int size);

}  // namespace BOOM
#endif  // BOOM_NEWLA_VECTOR_HPP

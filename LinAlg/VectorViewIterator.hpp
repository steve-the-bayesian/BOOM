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

#ifndef BOOM_VECTOR_VIEW_ITERATOR_HPP
#define BOOM_VECTOR_VIEW_ITERATOR_HPP

#include <cassert>
#include <iterator>
#include "uint.hpp"

namespace BOOM {
  class VectorViewIterator
      : public std::iterator<std::random_access_iterator_tag, double> {
    double *pos;
    double *beg;
    int stride;

   public:
    typedef VectorViewIterator VVI;
    VectorViewIterator(double *p, double *b, int s)
        : pos(p), beg(b), stride(s) {}
    VectorViewIterator(const VVI &rhs)
        : pos(rhs.pos), beg(rhs.beg), stride(rhs.stride) {}
    VVI &operator=(const VVI &rhs) {
      if (&rhs != this) {
        pos = rhs.pos;
        beg = rhs.beg;
        stride = rhs.stride;
      }
      return *this;
    }

    bool operator==(const VVI &rhs) const { return pos == rhs.pos; }
    bool operator!=(const VVI &rhs) const { return pos != rhs.pos; }

    double &operator*() const { return *pos; }
    VVI &operator++() {
      pos += stride;
      return *this;
    }
    VVI operator++(int) {
      VVI ans(*this);
      pos += stride;
      return ans;
    }
    VVI &operator--() {
      pos -= stride;
      return *this;
    }
    VVI operator--(int) {
      VVI ans(*this);
      pos -= stride;
      return ans;
    }

    double &operator[](uint n) const { return *(beg + n * stride); }
    VVI &operator+=(uint n) {
      pos += n * stride;
      return *this;
    }
    VVI &operator-=(uint n) {
      pos -= n * stride;
      return *this;
    }
    VVI operator+(uint n) const {
      VVI ans(*this);
      ans += n;
      return ans;
    }
    VVI operator-(uint n) const {
      VVI ans(*this);
      ans -= n;
      return ans;
    }
    std::ptrdiff_t operator-(const VVI &rhs) const {
      assert(stride == rhs.stride);
      return pos > rhs.pos ? (pos - rhs.pos) / stride
                           : (rhs.pos - pos) / stride;
    }
    bool operator<(const VVI &rhs) const { return pos < rhs.pos; }
    bool operator>(const VVI &rhs) const { return pos > rhs.pos; }
    bool operator<=(const VVI &rhs) const { return pos <= rhs.pos; }
    bool operator>=(const VVI &rhs) const { return pos >= rhs.pos; }
  };

  inline VectorViewIterator operator-(uint n, const VectorViewIterator &i) {
    return i - n;
  }

  //----------------------------
  class VectorViewConstIterator
      : public std::iterator<std::random_access_iterator_tag, double> {
    const double *pos;  // pos data values will not change
    const double *beg;  // beg pointer will not change
    int stride;

   public:
    typedef VectorViewConstIterator VVIC;
    VectorViewConstIterator(const double *p, const double *b, int s)
        : pos(p), beg(b), stride(s) {}
    VectorViewConstIterator(const VVIC &rhs)
        : pos(rhs.pos), beg(rhs.beg), stride(rhs.stride) {}
    VVIC &operator=(const VVIC &rhs) {
      if (&rhs != this) {
        pos = rhs.pos;
        beg = rhs.beg;
        stride = rhs.stride;
      }
      return *this;
    }

    bool operator==(const VVIC &rhs) const { return pos == rhs.pos; }
    bool operator!=(const VVIC &rhs) const { return pos != rhs.pos; }

    const double &operator*() const { return *pos; }
    VVIC &operator++() {
      pos += stride;
      return *this;
    }
    VVIC operator++(int) {
      VVIC ans(*this);
      pos += stride;
      return ans;
    }
    VVIC &operator--() {
      pos -= stride;
      return *this;
    }
    VVIC operator--(int) {
      VVIC ans(*this);
      pos -= stride;
      return ans;
    }

    const double &operator[](uint n) const { return *(beg + n * stride); }
    VVIC &operator+=(uint n) {
      pos += n * stride;
      return *this;
    }
    VVIC &operator-=(uint n) {
      pos -= n * stride;
      return *this;
    }
    VVIC operator+(uint n) const {
      VVIC ans(*this);
      ans += n;
      return ans;
    }
    VVIC operator-(uint n) const {
      VVIC ans(*this);
      ans -= n;
      return ans;
    }
    std::ptrdiff_t operator-(const VVIC &rhs) const {
      assert(stride == rhs.stride);
      return pos > rhs.pos ? (pos - rhs.pos) / stride
                           : (rhs.pos - pos) / stride;
    }
    bool operator<(const VVIC &rhs) const { return pos < rhs.pos; }
    bool operator>(const VVIC &rhs) const { return pos > rhs.pos; }
    bool operator<=(const VVIC &rhs) const { return pos <= rhs.pos; }
    bool operator>=(const VVIC &rhs) const { return pos >= rhs.pos; }
  };

}  // namespace BOOM
#endif  // BOOM_VECTOR_VIEW_ITERATOR_HPP
